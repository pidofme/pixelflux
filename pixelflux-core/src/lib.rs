/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Python-free contracts extracted from Pixelflux 2.0.0.
//!
//! This first maintained-fork split contains data and rate-control policy that both the PyO3
//! compatibility wrapper and future Rust-native consumers can share without linking Python. GPU,
//! compositor, frame-pool, and encoder modules remain in the preserved fork until their staged
//! extraction in Waydesk milestones 5R-C and 5R-D.

mod rate_control;
mod settings;

pub use rate_control::vbv_bits;
pub use settings::{LiveTunables, RustCaptureSettings};
