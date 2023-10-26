use std::fmt::Display;
use std::thread::sleep;
use crate::util::{ErrorKind, NmlError};
use rand::Rng;
use crate::util::ErrorKind::{InvalidCols, InvalidRows};


/// Nml_Matrix represents a matrix with a given number of rows and columns, the Data is stored in a one dimensional array using row-major-ordering
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
        match (num_rows * num_cols) as usize == data.len()  {
            false => Err(NmlError::new(ErrorKind::CreateMatrix)),
            true  => {
                let is_square = num_rows == num_cols;
                Ok(NmlMatrix {
                    num_rows,
                    num_cols,
                    data,
                    is_square,
                })},
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
            data,
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

    pub fn equality(self: &Self, matrix: &NmlMatrix) -> bool {
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
        match column < self.num_cols {
            false => Err(NmlError::new(ErrorKind::InvalidCols)),
            true => {
                let mut data: Vec<f64> = Vec::with_capacity(self.num_rows as usize);
                for i in 0..self.num_rows {
                    data.push(self.data[(i * self.num_rows + column) as usize]);
                }
                Ok(Self {
                    num_cols: 1,
                    num_rows: self.num_rows,
                    data,
                    is_square: false
                })
            },
        }
    }


    pub fn get_row(self: &Self, row: u32) -> Result<Self, NmlError> {
        match row < self.num_rows {
            true => {
                let data: Vec<f64> = self.data[(row * self.num_cols) as usize..(row * self.num_cols + self.num_cols) as usize].to_vec().clone();
                Ok(Self {
                    num_cols: self.num_cols,
                    num_rows: 1,
                    data,
                    is_square: false
                })
            },
            false => Err(NmlError::new(ErrorKind::InvalidRows)),
        }
    }
    /// Method sets the value of a given cell in the matrix through a mutable reference
    pub fn set_value(self: &mut Self, row: u32, col: u32, data: f64) -> Result<(), NmlError> {
        let valid_tuple: (bool, bool) = (row < self.num_rows, col < self.num_cols);
        match valid_tuple {
            (false, _) => Err(NmlError::new(ErrorKind::InvalidRows)),
            (_, false) => Err(NmlError::new(ErrorKind::InvalidCols)),
            (true, true) => {
                self.data[(row * self.num_cols + col) as usize] = data;
                Ok(())
            },
        }
    }
    /// Method sets the values of all cells ro a given value
    pub fn set_all_values(self: &mut Self, value: f64) {
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                self.data[(i * self.num_cols + j) as usize] = value;
            }
        }
    }
    ///checks if the matrix is square and sets the diagonal values to a given value
    pub fn set_dig_values(self: &mut Self, value: f64) -> Result<(), NmlError> {
        if self.is_square == true {
            for i in 0..self.num_rows {
                self.data[(i * self.num_cols + i) as usize] = value;
            }
        }
        match self.is_square {
            true => Ok(()),
            false => Err(NmlError::new(ErrorKind::MatrixNotSquare)),
        }
    }

    pub fn multiply_row_scalar(self: &mut Self, row: u32, scalar: f64) -> Result<(), NmlError> {
        match row < self.num_rows {
            false => Err(NmlError::new(ErrorKind::InvalidRows)),
            true => {
                for i in 0..self.num_cols {
                    self.data[(row * self.num_cols + i) as usize] *= scalar;
                }
                Ok(())
            },
        }
    }

    pub fn multiply_col_scalar(self: &mut Self, col: u32, scalar : f64) -> Result<(), NmlError>{
        match col < self.num_cols {
            false => Err(NmlError::new(ErrorKind::InvalidCols)),
            true => {
                for i in 0..self.num_rows {
                    self.data[(i * self.num_cols + col) as usize] *= scalar;

                }
                Ok(())
            }
        }
    }

    pub fn multiply_matrix_scalar(self: &mut Self, scalar: f64) {
        for i in 0..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    /// row_1 ist multiplied with scalar_1, this is analog for 2. row_1 will be modified with the solution (row_1 = row_1 * scalar + row_2 * scalar_2)
    pub fn add_rows(self: &mut Self, row_1: u32, scalar_1: f64, row_2: u32, scalar_2: f64) -> Result<(), NmlError>{
        match row_1 < self.num_rows && row_2 < self.num_rows {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                for i in 0..self.num_cols {
                    let value = self.data[(row_1 * self.num_cols + i) as usize];
                    self.data[(row_1 * self.num_cols + i) as usize] = value * scalar_1 + self.data[(row_2 * self.num_cols + i) as usize] * scalar_2;
                }
                Ok(())
            }
        }
    }

    pub fn remove_column(self: &Self, col: u32) -> Result<NmlMatrix, NmlError>{
        match col < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                let mut data: Vec<f64> = Vec::with_capacity(self.data.len());
                let indexes: Vec<usize> = (col as usize..self.data.len()).step_by(self.num_cols as usize).collect();
                for i in 0..self.data.len() {
                    if !indexes.contains(&i) {
                        data.push(self.data[i]);
                    }
                }
                Ok(
                    Self {
                        num_cols: 1,
                        num_rows: self.num_rows,
                        data,
                        is_square: false
                    }
                )
            }
        }
    }

    pub fn remove_row(self: &Self, row: u32) -> Result<NmlMatrix, NmlError>{
        match row < self.num_rows {
            false => Err(NmlError::new(ErrorKind::InvalidRows)),
            true => {
                let data: Vec<f64> = self.data[((row +1) * self.num_cols) as usize ..self.data.len()].to_vec();
                Ok(Self{
                    num_cols: self.num_cols,
                    num_rows: 1,
                    data,
                    is_square: false,
                })
            }
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





impl Eq for NmlMatrix {}
impl PartialEq for NmlMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.equality(other)
    }
}
