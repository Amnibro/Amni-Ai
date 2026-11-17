use std::collections::HashMap;

pub const MAX: usize = 8;

/// A plot of land.
#[derive(Debug, Clone)]
pub struct Plot {
    pub name: String,
    pub area: f32,
}

pub enum Crop {
    Bean,
    Potato,
}

pub trait Describe {
    fn describe(&self) -> String;
}

impl Describe for Plot {
    fn describe(&self) -> String {
        format!("{} ({} sq m)", self.name, self.area)
    }
}

pub fn total_area(plots: &[Plot]) -> f32 {
    plots.iter().map(|p| p.area).sum()
}

pub async fn load(path: &str) -> Option<HashMap<String, Plot>> {
    None
}

mod tests {
    use super::*;
}
