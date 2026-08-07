//! Helpers for setting up Python imports

macro_rules! register_submodule {
    ($parent_obj:expr, $parent_path:expr, $name:ident) => {{
        let py = $parent_obj.py();
        let full_name = format!("{}.{}", $parent_path, stringify!($name));
        py.import("sys")?
            .getattr("modules")?
            .set_item(&full_name, $parent_obj.getattr(stringify!($name))?)?;
    }};
}

pub(crate) use register_submodule;
