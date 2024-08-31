use {
    candle_core::{DType, Device, IndexOp, Module, Tensor},
    candle_nn::VarBuilder,
    candle_transformers::object_detection,
    image::{imageops, DynamicImage, GenericImageView},
    std::array,
};

pub mod yolo_v8;

pub const POINT_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

pub struct PoseDetection {
    confidence_threshold: f32,
    device: Device,
    model: yolo_v8::YoloV8Pose,
    non_maximum_suppression_threshold: f32,
}

impl PoseDetection {
    /// Fetch `yolov8n-pose.safetensors` from `lmz/candle-yolo-v8` and load it.
    pub fn fetch_and_load() -> anyhow::Result<Self> {
        let hf = hf_hub::api::sync::Api::new()?;
        let repo = hf.model(String::from("lmz/candle-yolo-v8"));
        let model_path = repo.get("yolov8s-pose.safetensors")?;

        let device = Device::cuda_if_available(0)?;
        let vars =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

        let model = yolo_v8::YoloV8Pose::load(vars, yolo_v8::Multiples::s(), 1, (17, 3))?;

        tracing::info!("model loaded");

        Ok(Self {
            confidence_threshold: 0.25,
            device,
            model,
            non_maximum_suppression_threshold: 0.45,
        })
    }

    /// Run pose detection on the provided `image`.
    pub fn detect(&self, image: DynamicImage) -> anyhow::Result<Vec<Pose>> {
        tracing::info!("running detection");

        let image: DynamicImage = image.into_rgb8().into();
        let (width, height) = image.dimensions();
        let (width, height) = if width < height {
            let width = width * 640 / height;

            (width.saturating_sub(31).next_multiple_of(32), 640)
        } else {
            let height = height * 640 / width;

            (640, height.saturating_sub(31).next_multiple_of(32))
        };

        let data = image
            .resize_exact(width, height, imageops::CatmullRom)
            .into_rgb8()
            .into_raw();

        let input = (Tensor::from_vec(data, (height as usize, width as usize, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?
            * (1.0 / 255.0))?;

        let predictions = self
            .model
            .forward(&input)?
            .squeeze(0)?
            .to_device(&Device::Cpu)?;

        let (predictions_size, predictions_len) = predictions.dims2()?;

        if predictions_size != 17 * 3 + 4 + 1 {
            return Err(anyhow::anyhow!("unexpected output size"));
        }

        let mut poses = Vec::new();

        for index in 0..predictions_len {
            let prediction = Vec::<f32>::try_from(predictions.i((.., index))?)?;
            let confidence = prediction[4];

            if confidence > self.confidence_threshold {
                let points = array::from_fn::<_, 16, _>(|index| object_detection::KeyPoint {
                    x: prediction[3 * index + 5],
                    y: prediction[3 * index + 6],
                    mask: prediction[3 * index + 7],
                });

                poses.push(object_detection::Bbox {
                    xmin: prediction[0] - prediction[2] / 2.0,
                    ymin: prediction[1] - prediction[3] / 2.0,
                    xmax: prediction[0] + prediction[2] / 2.0,
                    ymax: prediction[1] + prediction[3] / 2.0,
                    confidence,
                    data: points,
                });
            }
        }

        let mut poses = vec![poses];

        object_detection::non_maximum_suppression(
            &mut poses,
            self.non_maximum_suppression_threshold,
        );

        let x_scale = image.width() as f32 / width as f32;
        let y_scale = image.height() as f32 / height as f32;

        let poses = poses.remove(0);
        let mut scaled = Vec::with_capacity(poses.len());

        for pose in poses {
            let x = (pose.xmin * x_scale).round() as i32;
            let y = (pose.ymin * y_scale).round() as i32;
            let width = ((pose.xmax - pose.xmin) * x_scale).round();
            let height = ((pose.ymax - pose.ymin) * y_scale).round();

            if width < 0.0 || height < 0.0 {
                continue;
            }

            let points = pose
                .data
                .into_iter()
                .map(|point| Point {
                    x: point.x * x_scale,
                    y: point.y * y_scale,
                    mask: point.mask,
                })
                .collect();

            scaled.push(Pose {
                x,
                y,
                width: width as u32,
                height: height as u32,
                points,
            })
        }

        tracing::info!("detected {scaled:?}");

        Ok(scaled)
    }
}

#[derive(Clone, Debug)]
pub struct Pose {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub points: Vec<Point>,
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub mask: f32,
}
