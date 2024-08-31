use {
    image::{DynamicImage, ImageBuffer},
    imageproc::{drawing, rect::Rect},
    rusttype::{Font, Scale},
    sctk::{
        compositor::{CompositorHandler, CompositorState},
        output::{OutputHandler, OutputState},
        registry::{ProvidesRegistryState, RegistryState},
        seat::{
            keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers},
            pointer::{PointerEvent, PointerHandler},
            Capability, SeatHandler, SeatState,
        },
        shell::{
            wlr_layer::{
                Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
                LayerSurfaceConfigure,
            },
            WaylandSurface,
        },
        shm::{slot::SlotPool, Shm, ShmHandler},
    },
    std::{
        borrow::Cow,
        mem,
        process::{Command, Stdio},
        sync::mpsc,
        thread,
    },
    wayland_client::{
        globals,
        protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface},
        Connection, EventQueue, QueueHandle,
    },
};

pub mod pose_detection;

pub struct Clyde {
    event_queue: EventQueue<Overlay>,
    overlay: Overlay,
}

pub struct Overlay {
    exit: bool,
    first_draw: bool,
    font: Font<'static>,
    height: u32,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    keyboard_focus: bool,
    layer: LayerSurface,
    memory_pool: SlotPool,
    output: OutputState,
    pointer: Option<wl_pointer::WlPointer>,
    poses: Vec<pose_detection::Pose>,
    poses_receiver: mpsc::Receiver<Vec<pose_detection::Pose>>,
    registry: RegistryState,
    seat: SeatState,
    shared_memory: Shm,
    shift: Option<u32>,
    width: u32,
}

impl Clyde {
    pub fn new() -> anyhow::Result<Self> {
        let (poses_sender, poses_receiver) = mpsc::channel();

        thread::spawn(move || -> anyhow::Result<()> {
            let pose_detection = pose_detection::PoseDetection::fetch_and_load()?;

            loop {
                let image = screen_capture()?;

                poses_sender.send(pose_detection.detect(image)?)?;
            }
        });

        // Connect to the Wayland server (compositor).
        let connection = Connection::connect_to_env()?;

        // Obtain a list of globals the server advertises.
        let (globals, event_queue) = globals::registry_queue_init(&connection)?;
        let queue_handle = event_queue.handle();

        // For configuring surfaces.
        let compositor = CompositorState::bind(&globals, &queue_handle)?;

        // For creating a surface on a layer (overlay, wallpaper, etc).
        let layer_shell = LayerShell::bind(&globals, &queue_handle)?;

        // For shared memory buffers with the Wayland server (compositor).
        let shared_memory = Shm::bind(&globals, &queue_handle)?;

        // Create a surface.
        let surface = compositor.create_surface(&queue_handle);

        // Create a layer on the top (overlay), with an associated surface.
        let layer = layer_shell.create_layer_surface(
            &queue_handle,
            surface,
            Layer::Top,
            Some("clyde"),
            None,
        );

        // Anchor to the top-left.
        layer.set_anchor(Anchor::TOP | Anchor::LEFT);

        // Don't handle keyboard input.
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);

        // The same as the screen size.
        layer.set_size(2560, 1440);

        // Apparently you have to inform the Wayland server about a layer then configure it again later with the correct options.
        layer.commit();

        // Width (2560) * height (1440) * 8-bit RGBA (4).
        let memory_pool = SlotPool::new(2560 * 1440 * 4, &shared_memory)?;

        // Overlay text.
        let font = Font::try_from_bytes(include_bytes!("regular.ttf"))
            .ok_or_else(|| anyhow::anyhow!("failed to load font"))?;

        let overlay = Overlay {
            exit: false,
            first_draw: true,
            font,
            height: 1440,
            keyboard: None,
            keyboard_focus: false,
            layer,
            memory_pool,
            output: OutputState::new(&globals, &queue_handle),
            pointer: None,
            poses_receiver,
            poses: Vec::new(),
            registry: RegistryState::new(&globals),
            seat: SeatState::new(&globals, &queue_handle),
            shared_memory,
            shift: None,
            width: 2560,
        };

        Ok(Self {
            event_queue,
            overlay,
        })
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        let Self {
            event_queue,
            overlay,
        } = self;

        while !overlay.exit {
            event_queue.blocking_dispatch(overlay)?;
        }

        Ok(())
    }
}

impl Overlay {
    pub fn render(&mut self, queue_handle: &QueueHandle<Self>) {
        let _ = self.render_inner(queue_handle);
    }

    fn render_inner(&mut self, queue_handle: &QueueHandle<Self>) -> anyhow::Result<()> {
        let Self {
            font,
            height,
            layer,
            poses,
            poses_receiver,
            width,
            ..
        } = self;

        let stride = (*width * 4) as i32;
        let (buffer, canvas) = self.memory_pool.create_buffer(
            *width as i32,
            *height as i32,
            stride,
            wl_shm::Format::Argb8888,
        )?;

        // Clear the previously drawn pixels.
        canvas.fill(0);

        // Obtain a more convenient view of the buffer data.
        let mut image = ImageBuffer::<image::Rgba<u8>, _>::from_raw(*width, *height, canvas)
            .ok_or_else(|| anyhow::anyhow!("buffer size does not match"))?;

        // Update the current rendered poses if any are available, otherwise, don't do anything.
        if let Ok(new_poses) = poses_receiver.try_recv() {
            *poses = new_poses;
        }

        let box_color = image::Rgba([255, 255, 255, 0]);
        let head_dot_color = image::Rgba([255, 0, 0, 255]);
        let dot_color = image::Rgba([255, 255, 255, 0]);
        let line_color = image::Rgba([70, 255, 255, 255]);
        let text_color = image::Rgba([255, 255, 255, 255]);
        let text_shadow_color = image::Rgba([0, 0, 0, 255]);

        let text = if poses.is_empty() {
            Cow::Borrowed("Nothing detected")
        } else {
            Cow::Owned(format!("Detected {}", poses.len()))
        };

        // Text shadow.
        drawing::draw_text_mut(
            &mut image,
            text_shadow_color,
            2,
            2,
            Scale::uniform(24.0),
            font,
            &text,
        );

        // Text itself.
        drawing::draw_text_mut(
            &mut image,
            text_color,
            0,
            0,
            Scale::uniform(24.0),
            font,
            &text,
        );

        // Draw poses.
        for pose in poses.iter() {
            let pose_detection::Pose {
                x,
                y,
                width,
                height,
                points,
            } = pose;

            drawing::draw_hollow_rect_mut(
                &mut image,
                Rect::at(*x, *y).of_size(*width, *height),
                box_color,
            );

            for (index, point) in points.iter().copied().enumerate() {
                let dot_color = if index == 0 {
                    head_dot_color
                } else {
                    dot_color
                };

                if point.mask < 0.6 {
                    continue;
                }

                let point = (point.x.round() as i32, point.y.round() as i32);

                drawing::draw_filled_circle_mut(&mut image, point, 2, dot_color);
            }

            for (from, to) in pose_detection::POINT_CONNECTIONS {
                let Some(from) = points.get(from) else {
                    continue;
                };

                let Some(to) = points.get(to) else {
                    continue;
                };

                if from.mask < 0.6 || to.mask < 0.6 {
                    continue;
                }

                let from = (from.x, from.y);
                let to = (to.x, to.y);

                drawing::draw_line_segment_mut(&mut image, from, to, line_color);
            }
        }

        // Invalidate the layer.
        layer
            .wl_surface()
            .damage_buffer(0, 0, *width as i32, *height as i32);

        // Request the next frame.
        layer
            .wl_surface()
            .frame(queue_handle, layer.wl_surface().clone());

        // Attach the buffer to the layer.
        buffer.attach_to(layer.wl_surface())?;

        // Commit the layer to present.
        layer.commit();

        Ok(())
    }
}

impl CompositorHandler for Overlay {
    fn scale_factor_changed(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _connection: &Connection,
        queue_handle: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        self.render(queue_handle);
    }
}

impl OutputHandler for Overlay {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output
    }

    fn new_output(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for Overlay {
    fn closed(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _layer: &LayerSurface,
    ) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _connection: &Connection,
        queue_handle: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let Self {
            width,
            height,
            first_draw,
            ..
        } = self;

        if configure.new_size.0 == 0 || configure.new_size.1 == 0 {
            *width = 2560;
            *height = 1440;
        } else {
            *width = configure.new_size.0;
            *height = configure.new_size.1;
        }

        if mem::take(first_draw) {
            self.render(queue_handle);
        }
    }
}

impl SeatHandler for Overlay {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat
    }

    fn new_seat(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _seat: wl_seat::WlSeat,
    ) {
    }

    fn new_capability(
        &mut self,
        _connection: &Connection,
        queue_handle: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        return;

        if capability == Capability::Keyboard && self.keyboard.is_none() {
            let keyboard = self
                .seat
                .get_keyboard(queue_handle, &seat, None)
                .expect("Failed to create keyboard");

            self.keyboard = Some(keyboard);
        }

        if capability == Capability::Pointer && self.pointer.is_none() {
            let pointer = self
                .seat
                .get_pointer(queue_handle, &seat)
                .expect("Failed to create pointer");

            self.pointer = Some(pointer);
        }
    }

    fn remove_capability(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard && self.keyboard.is_some() {
            println!("Unset keyboard capability");
            self.keyboard.take().unwrap().release();
        }

        if capability == Capability::Pointer && self.pointer.is_some() {
            println!("Unset pointer capability");
            self.pointer.take().unwrap().release();
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for Overlay {
    fn enter(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _serial: u32,
        _raw: &[u32],
        _keysyms: &[Keysym],
    ) {
        if self.layer.wl_surface() == surface {
            self.keyboard_focus = true;
        }
    }

    fn leave(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _serial: u32,
    ) {
        if self.layer.wl_surface() == surface {
            self.keyboard_focus = false;
        }
    }

    fn press_key(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        event: KeyEvent,
    ) {
        if event.keysym == Keysym::Escape {
            self.exit = true;
        }
    }

    fn release_key(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _keyoard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        _event: KeyEvent,
    ) {
    }

    fn update_modifiers(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        _modifiers: Modifiers,
    ) {
    }
}

impl PointerHandler for Overlay {
    fn pointer_frame(
        &mut self,
        _connection: &Connection,
        _queue_handle: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        _events: &[PointerEvent],
    ) {
    }
}

impl ShmHandler for Overlay {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shared_memory
    }
}

impl ProvidesRegistryState for Overlay {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }

    sctk::registry_handlers![OutputState, SeatState];
}

sctk::delegate_compositor!(Overlay);
sctk::delegate_keyboard!(Overlay);
sctk::delegate_layer!(Overlay);
sctk::delegate_output!(Overlay);
sctk::delegate_pointer!(Overlay);
sctk::delegate_registry!(Overlay);
sctk::delegate_seat!(Overlay);
sctk::delegate_shm!(Overlay);

fn screen_capture() -> anyhow::Result<DynamicImage> {
    tracing::info!("capturing screen");

    let bytes = Command::new("grim")
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()?
        .stdout;

    let image = image::load_from_memory(&bytes)?;

    tracing::info!("captured {}x{} image", image.width(), image.height());

    Ok(image)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut clyde = Clyde::new()?;

    clyde.run()?;

    Ok(())
}
