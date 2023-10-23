use std::fmt::Display;
use crate::util::{ErrorKind, NmlError};
use rand::Rng;


/// Nml_Matrix represents a matrix with a given number of rows and columns, the Data is stored in a one dimensonal array using row-major-ordering
pub struct NmlMatrix {
    pub num_rows: u32,
    pub num_cols: u32,
    //data is stored using row-major-ordering data[i][j] = data_new[i * m +j]
    pub data: Vec<f64>,
    pub is_square: bool,
}

impl NmlMatrix {

    //creates a matrix with a given Array of numbers
    pub fn new(num_rows: u32, num_cols: u32) -> Self {
        let data: Vec<f64> = Vec::with_capacity((num_rows * num_cols) as usize);
        Self {
            num_rows,
            num_cols,
            data,
            is_square: num_rows == num_cols,
        }
    }

    pub fn new_with_data(num_rows: u32, num_cols: u32, data: Vec<f64>) -> Result<Self, NmlError> {
        let valid = match data.len() == (num_cols * num_rows) as usize {
            false => Err(()),
            true  => Ok(()),
        };
        match valid {
            Ok(()) => {
                let is_square = num_rows == num_cols;
                Ok(NmlMatrix {
                    num_rows,
                    num_cols,
                    data,
                    is_square,
                })
            },
            Err(()) => Err(NmlError::new(ErrorKind::CreateMatrix)),
        }
    }


    pub fn nml_mat_rnd(num_rows: u32, num_cols: u32, minimum: f64, maximum: f64) -> Self {
        let mut rng = rand::thread_rng();
        let random_numbers: Vec<f64> = (0..100).map(|_| (rng.gen_range(minimum..maximum))).collect();
        Self {
            num_rows,
            num_cols,
            data: random_numbers,
            is_square: num_rows == num_cols,
        }
    }

    //creates a square matrix filled with 0.0
    pub fn nml_mat_sqr(size: u32) -> Self {
        Self {
            num_rows: size,
            num_cols: size,
            data: vec![0.0; (size * size) as usize],
            is_square: true,
        }
    }

    //creates a identity matrix with the given size
    pub fn nml_mat_eye(size: u32) -> Self {
        let mut data: Vec<f64> = vec![0.0; (size * size) as usize];
        for i in 0..size {
            data[(i * size + i) as usize] = 1.0;
        }
        Self {
            num_rows: size,
            num_cols: size,
            data: data,
            is_square: true,
        }
    }
    //uses only the reference to the matrix, so that the matrix is not moved
    pub fn nml_mat_cp(matrix: &NmlMatrix) -> Self {
        Self {
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            data: matrix.data.clone(),
            is_square: matrix.is_square,
        }
    }

    pub fn nml_mat_fromfile() -> Self {
        unimplemented!("Not implemented yet")
    }

    pub fn equality(self: &Self, matrix: NmlMatrix) -> bool {
        if self.num_rows != matrix.num_rows || self.num_cols != matrix.num_cols {
            return false;
        }
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                if self.data[(i * self.num_cols + j) as usize] != matrix.data[(i * matrix.num_cols + j) as usize] {
                    return false;
                }
            }
        }
        true
    }

    pub fn equality_in_tolerance(self: &Self, matrix: NmlMatrix, tolerance: f64) -> bool {
        if self.num_rows != matrix.num_rows || self.num_cols != matrix.num_cols {
            return false;
        }
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                if (self.data[(i * self.num_cols + j) as usize] - matrix.data[(i * matrix.num_cols + j) as usize]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    pub fn get_column(self: &Self, column: u32) -> Result<Self, NmlError> {
        // uses two match statements to first check if the given column number is valid
        // and then in the second match return either the Column/Matrix or the matrix-error
        let valid = match column < self.num_cols {
            true => Ok(()),
            false => Err(()),
        };
        match valid {
            Err(())=> Err(NmlError::new(ErrorKind::InvalidCols)),
            Ok(()) => {
                let mut data: Vec<f64> = Vec::with_capacity(self.num_rows as usize);
                for i in 0..self.num_rows {
                    data.push(self.data[(i * self.num_rows + column) as usize]);
                }
                Ok(Self {
                    num_cols: 1,
                    num_rows: self.num_rows,
                    data: data,
                    is_square: false
                })
            },
        }
    }


}

impl Display for NmlMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                output.push_str(&self.data[(i * self.num_cols + j) as usize].to_string());
                output.push_str(" ");
            }
            output.push_str("\n");
        }
        write!(f, "{}", output)
    }
}
