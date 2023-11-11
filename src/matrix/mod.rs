use std::fmt::Display;
use std::ops::Add;
use std::ptr::eq;
use crate::util::{ErrorKind, NmlError};
use rand::Rng;
use crate::util::ErrorKind::{CreateMatrix, InvalidCols, InvalidRows};


/// Nml_Matrix represents a matrix with a given number of rows and columns, the Data is stored in a one dimensional array using row-major-ordering (data[i][j] = data_new[i * m +j])
/// The Library contains a few methods to create matrices with or without data.
pub struct NmlMatrix {
    pub num_rows: u32,
    pub num_cols: u32,
    pub data: Vec<f64>,
    pub is_square: bool,
}

impl NmlMatrix {

    ///creates a matrix without data and reserves the capacity for the Data Vector
    pub fn new(num_rows: u32, num_cols: u32) -> Self {
        let data: Vec<f64> = Vec::with_capacity((num_rows * num_cols) as usize);
        Self {
            num_rows,
            num_cols,
            data,
            is_square: num_rows == num_cols,
        }
    }

    ///Constructor that uses a vector to initialize the matrix. checks if the entered rows and columns fit the vector size
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

    ///Returns a matrix with defined size and random data between minimum and maximum values
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

    ///Creates a square matrix of a given size, where all cells are filled with 0.0
    pub fn nml_mat_sqr(size: u32) -> Self {
        Self {
            num_rows: size,
            num_cols: size,
            data: vec![0.0; (size * size) as usize],
            is_square: true,
        }
    }

    ///creates a identity matrix with the given size
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
    ///Creates a new matrix which is a copy of a given matrix. Uses only the reference to the matrix, so that the original matrix is not moved
    pub fn nml_mat_cp(matrix: &NmlMatrix) -> Self {
        Self {
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            data: matrix.data.clone(),
            is_square: matrix.is_square,
        }
    }
    ///Unimplemented method that should read in a matrix from a file
    pub fn nml_mat_from_file() -> Self {
        unimplemented!("Not implemented yet")
    }
    ///Checks if two matrices are equal by checking their dimensions and after that every cell. Method is also used for implementation of trait PatialEq
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
    ///Checks if two matrices are equal with a given tolerance
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
    ///Returns a result with a specified Column of a matrix, which in itself also is a matrix. If the specified matrix is not in the matrix the result will contain an error
    pub fn get_column(self: &Self, column: u32) -> Result<Self, NmlError> {
        match column < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
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

    ///Returns a result which either contains a row of a matrix (which is also a matrix) or a error
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
            false => Err(NmlError::new(InvalidRows)),
        }
    }
    /// Method sets the value of a given cell in the matrix through a mutable reference
    pub fn set_value(self: &mut Self, row: u32, col: u32, data: f64) -> Result<(), NmlError> {
        let valid_tuple: (bool, bool) = (row < self.num_rows, col < self.num_cols);
        match valid_tuple {
            (false, _) => Err(NmlError::new(InvalidRows)),
            (_, false) => Err(NmlError::new(InvalidCols)),
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
    ///multiplies a given row with a given scalar in place. If the row-index is not in the matrix the returned Result will contain an error
    pub fn multiply_row_scalar(self: &mut Self, row: u32, scalar: f64) -> Result<(), NmlError> {
        match row < self.num_rows {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                for i in 0..self.num_cols {
                    self.data[(row * self.num_cols + i) as usize] *= scalar;
                }
                Ok(())
            },
        }
    }
    ///multiplies a given column with a given scalar in place. If the row-index is not in the matrix the returned Result will contain an error
    pub fn multiply_col_scalar(self: &mut Self, col: u32, scalar : f64) -> Result<(), NmlError>{
        match col < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                for i in 0..self.num_rows {
                    self.data[(i * self.num_cols + col) as usize] *= scalar;

                }
                Ok(())
            }
        }
    }
    ///multiplies the matrix in place with a given scalar
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
    ///Method that swaps two given rows of a matrix object in place. Returns either nothing or an NmlError if the specified rows are not in the matrix
    pub fn swap_rows(self: &mut Self, row_1: u32, row_2: u32) -> Result<(), NmlError>{
        match row_1 < self.num_rows && row_2 < self.num_rows {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                for i in 0..self.num_cols {
                    let temp = self.data[(row_1 * self.num_cols + i) as usize];
                    self.data[(row_1 * self.num_cols + i) as usize] = self.data[(row_2 * self.num_cols + i) as usize];
                    self.data[(row_2 * self.num_cols + i) as usize] = temp;
                }
                Ok(())
            }
        }
    }
    ///Method that swaps two given rows of a matrix object in place. Returns either nothing or an NmlError if the specified rows are not in the matrix
    pub fn swap_columns(self: &mut Self, col_1: u32, col_2: u32) -> Result<(), NmlError>{
        match col_1 < self.num_cols && col_2 < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                for i in 0..self.num_rows {
                    let temp = self.data[(i*self.num_cols + col_1) as usize];
                    self.data[(i*self.num_cols + col_1) as usize] = self.data[(i*self.num_cols + col_2) as usize];
                    self.data[(i*self.num_cols + col_2) as usize] = temp;
                }
                Ok(())
            }
        }
    }
    ///Tries to remove a column of a matrix and returns the rest of the matrix as a now on. Does not move the original matrix.
    ///If the column is not in the original matrix the result will return an error
    pub fn remove_column(self: &Self, col: u32) -> Result<Self, NmlError>{
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
    ///Tries to remove a column of a matrix and returns the rest of the matrix as a now on. Does not move the original matrix.
    ///If the column is not in the original matrix the result will return an error
    pub fn remove_row(self: &Self, row: u32) -> Result<Self, NmlError>{
        match row < self.num_rows {
            false => Err(NmlError::new(InvalidRows)),
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

impl Add for NmlMatrix {
    type Output = Result<Self, NmlError>;

    fn add(self, rhs: Self) -> Self::Output {
        match self.data.len() == rhs.data.len() && self.num_cols == rhs.num_cols{
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<f64> = Vec::new();
                for i in 0..self.data.len() {
                    data.insert(i, self.data[i] + rhs.data[i]);
                }
                Ok(Self{
                    num_cols: self.num_cols,
                    num_rows: self.num_rows,
                    data,
                    is_square: self.is_square
                })
            }
        }
    }
}
