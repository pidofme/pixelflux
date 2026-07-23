/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Capture settings shared by Rust-native and PyO3 Pixelflux frontends.

/// The full set of capture and encode parameters passed to a Pixelflux backend.
#[derive(Clone, Debug, PartialEq)]
pub struct RustCaptureSettings {
    pub width: i32,
    pub height: i32,
    pub scale: f64,
    pub capture_x: i32,
    pub capture_y: i32,
    pub target_fps: f64,
    pub jpeg_quality: i32,
    pub paint_over_jpeg_quality: i32,
    pub use_paint_over_quality: bool,
    pub paint_over_trigger_frames: u32,
    pub damage_block_threshold: u32,
    pub damage_block_duration: u32,
    pub output_mode: i32,
    pub video_crf: i32,
    pub video_paintover_crf: i32,
    pub video_paintover_burst_frames: i32,
    pub video_fullcolor: bool,
    pub video_fullframe: bool,
    pub video_streaming_mode: bool,
    pub capture_cursor: bool,
    /// Longest cursor edge delivered out of band; `<= 0` means uncapped.
    pub cursor_size_cap: i32,
    pub watermark_path: String,
    pub watermark_location_enum: i32,
    pub encode_node_index: i32,
    pub use_cpu: bool,
    pub use_openh264: bool,
    pub debug_logging: bool,
    pub auto_adjust_screen_capture_size: bool,
    pub recording_socket: String,
    /// Emit raw payloads without Pixelflux per-stripe headers.
    pub omit_stripe_headers: bool,
    pub video_cbr_mode: bool,
    pub video_bitrate_kbps: i32,
    /// CBR VBV/HRD size as a multiple of one frame's bit budget.
    pub video_vbv_multiplier: f64,
    /// Seconds between scheduled recovery keyframes; `<= 0` keeps an infinite GOP.
    pub keyframe_interval_s: f64,
    /// Encoder QP clamps; zero retains the encoder default.
    pub video_min_qp: i32,
    pub video_max_qp: i32,
}

/// The subset of settings that may be updated without recreating a capture session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveTunables {
    pub jpeg_quality: i32,
    pub paint_over_jpeg_quality: i32,
    pub use_paint_over_quality: bool,
    pub paint_over_trigger_frames: u32,
    pub video_crf: i32,
    pub video_paintover_crf: i32,
    pub video_paintover_burst_frames: i32,
    pub video_streaming_mode: bool,
    pub keyframe_interval_s: f64,
    pub capture_cursor: bool,
}

impl LiveTunables {
    /// Snapshot the live-tunable subset from full settings.
    #[must_use]
    pub fn from_settings(settings: &RustCaptureSettings) -> Self {
        Self {
            jpeg_quality: settings.jpeg_quality,
            paint_over_jpeg_quality: settings.paint_over_jpeg_quality,
            use_paint_over_quality: settings.use_paint_over_quality,
            paint_over_trigger_frames: settings.paint_over_trigger_frames,
            video_crf: settings.video_crf,
            video_paintover_crf: settings.video_paintover_crf,
            video_paintover_burst_frames: settings.video_paintover_burst_frames,
            video_streaming_mode: settings.video_streaming_mode,
            keyframe_interval_s: settings.keyframe_interval_s,
            capture_cursor: settings.capture_cursor,
        }
    }

    /// Apply the live-tunable subset to full settings in place.
    pub fn apply_to(self, settings: &mut RustCaptureSettings) {
        settings.jpeg_quality = self.jpeg_quality;
        settings.paint_over_jpeg_quality = self.paint_over_jpeg_quality;
        settings.use_paint_over_quality = self.use_paint_over_quality;
        settings.paint_over_trigger_frames = self.paint_over_trigger_frames;
        settings.video_crf = self.video_crf;
        settings.video_paintover_crf = self.video_paintover_crf;
        settings.video_paintover_burst_frames = self.video_paintover_burst_frames;
        settings.video_streaming_mode = self.video_streaming_mode;
        settings.keyframe_interval_s = self.keyframe_interval_s;
        settings.capture_cursor = self.capture_cursor;
    }
}

impl Default for RustCaptureSettings {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            scale: 1.0,
            capture_x: 0,
            capture_y: 0,
            target_fps: 60.0,
            jpeg_quality: 75,
            paint_over_jpeg_quality: 95,
            use_paint_over_quality: true,
            paint_over_trigger_frames: 15,
            damage_block_threshold: 10,
            damage_block_duration: 30,
            output_mode: 0,
            video_crf: 25,
            video_paintover_crf: 18,
            video_paintover_burst_frames: 5,
            video_fullcolor: false,
            video_fullframe: false,
            video_streaming_mode: false,
            capture_cursor: false,
            cursor_size_cap: 32,
            watermark_path: String::new(),
            watermark_location_enum: 0,
            encode_node_index: -2,
            use_cpu: false,
            use_openh264: false,
            debug_logging: false,
            auto_adjust_screen_capture_size: false,
            recording_socket: String::new(),
            omit_stripe_headers: false,
            video_cbr_mode: false,
            video_bitrate_kbps: 4000,
            video_vbv_multiplier: 0.0,
            keyframe_interval_s: 0.0,
            video_min_qp: 0,
            video_max_qp: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pixelflux_2_0_0() {
        let settings = RustCaptureSettings::default();
        assert_eq!((settings.width, settings.height), (1024, 768));
        assert_eq!(settings.target_fps, 60.0);
        assert_eq!(settings.jpeg_quality, 75);
        assert_eq!(settings.video_crf, 25);
        assert_eq!(settings.video_bitrate_kbps, 4000);
        assert_eq!(settings.encode_node_index, -2);
        assert_eq!(settings.cursor_size_cap, 32);
    }

    #[test]
    fn live_tunables_only_change_the_live_subset() {
        let mut settings = RustCaptureSettings {
            width: 1920,
            height: 1080,
            jpeg_quality: 60,
            video_crf: 31,
            capture_cursor: true,
            ..RustCaptureSettings::default()
        };
        let original_width = settings.width;
        let original_height = settings.height;
        let mut tunables = LiveTunables::from_settings(&settings);
        tunables.jpeg_quality = 88;
        tunables.video_crf = 19;
        tunables.capture_cursor = false;
        tunables.apply_to(&mut settings);

        assert_eq!(
            (settings.width, settings.height),
            (original_width, original_height)
        );
        assert_eq!(settings.jpeg_quality, 88);
        assert_eq!(settings.video_crf, 19);
        assert!(!settings.capture_cursor);
    }
}
