use crate::util::{ErrorKind, NmlError};



pub struct NmlMatrix {
    num_rows: u32,
    num_cols: u32,
    //data is stored using row-major-ordering data[i][j] = data_new[i * m +j]
    data: Vec<f64>,
    is_square: bool,
}

impl NmlMatrix {
    pub fn new(num_rows: u32, num_cols: u32) -> Result<Self, ErrorKind> {
        let size = num_rows * num_cols;
        let valid = num_rows > 0 && num_cols > 0;
        let valid = match valid {
            false => NmlError::new(ErrorKind::InvalidRows),
            true  => Ok(()),
        };
        match valid {
            Ok(()) => {
                let is_square = num_rows == num_cols;
                let data:Vec<f64> = vec![0.0; size as usize];
                Ok(NmlMatrix {
                    num_rows,
                    num_cols,
                    data,
                    is_square,
                })
            },
            Err(ErrorKind::CreateMatrix) => NmlError::new(ErrorKind::CreateMatrix),
        }
    }

    pub fn nml_mat_rnd(num_rows: u32, num_cols: u32, ) -> Self {

    }

    pub fn nml_mat_sqr() -> Self {

    }

    pub fn nml_mat_eye() -> Self {

    }

    pub fn nml_mat_cp() -> Self {

    }

    pub fn nml_mat_fromfile() -> Self {

    }
}
