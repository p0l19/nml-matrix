use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use num_traits::{Num, Signed};
use rand::distributions::uniform::SampleUniform;
use crate::util::{ErrorKind, NmlError};
use rand::Rng;
use crate::util::ErrorKind::{CreateMatrix, InvalidCols, InvalidRows};


/// Nml_Matrix represents a matrix with a given number of rows and columns, the Data is stored in a one dimensional array using row-major-ordering (data[i][j] = data_new[i * m +j], where m is the number of columns)
/// The Library contains a few methods to create matrices with or without data.
#[derive(Debug)]
pub struct NmlMatrix<T> {
    pub num_rows: u32,
    pub num_cols: u32,
    pub data: Box<[T]>,
    pub is_square: bool,
}

impl<T> NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign + Display + SampleUniform {
    ///creates a matrix without data and reserves the capacity for the Data Vector
    pub fn new(num_rows: u32, num_cols: u32) -> NmlMatrix<T> {
        let data: Box<[T]> = vec![T::default(); (num_rows * num_cols) as usize].into_boxed_slice();
        Self {
            num_rows,
            num_cols,
            data,
            is_square: num_rows == num_cols,
        }
    }
    ///use a 2d Vector to initialize the matrix. Each Vector in the 2d Vector is a row and the length of these vectors are the columns
    pub fn new_with_2d_vec(num_rows: u32, num_cols: u32, data_2d: &mut Vec<Vec<T>>) -> Result<Self, NmlError> {
        let rows: u32 = data_2d.len() as u32;
        let mut cols_match = true;
        let mut vec_data: Vec<T> = Vec::with_capacity((num_rows*num_cols) as usize);
        for element in data_2d {
            if element.len() as u32 != num_cols {
                cols_match = false;
                break;
            }
            vec_data.append(element);
        }
        let data: Box<[T]> = vec_data.into_boxed_slice();
        match cols_match && rows == num_rows{
            true => {Ok(Self{
                num_cols,
                num_rows,
                data,
                is_square: num_rows == num_rows
            })},
            false => {Err(NmlError::new(ErrorKind::CreateMatrix))}
        }
    }

    ///Constructor that uses a vector to initialize the matrix. checks if the entered rows and columns fit the vector size
    pub fn new_with_data(num_rows: u32, num_cols: u32, data: Box<[T]>) -> Result<Self, NmlError> {
        match (num_rows * num_cols) as usize == data.len() {
            false => Err(NmlError::new(ErrorKind::CreateMatrix)),
            true => {
                let is_square = num_rows == num_cols;
                Ok(NmlMatrix {
                    num_rows,
                    num_cols,
                    data,
                    is_square,
                })
            },
        }
    }

    ///Returns a matrix with defined size and random data between minimum and maximum values
    pub fn nml_mat_rnd(num_rows: u32, num_cols: u32, minimum: T, maximum: T) -> Self {
        let mut rng = rand::thread_rng();
        let random_numbers: Vec<T> = (0..100).map(|_| (rng.gen_range(minimum..maximum))).collect();
        Self {
            num_rows,
            num_cols,
            data: random_numbers.into_boxed_slice(),
            is_square: num_rows == num_cols,
        }
    }

    ///Creates a square matrix of a given size, where all cells are filled with 0.0
    pub fn nml_mat_sqr(size: u32) -> Self {
        Self {
            num_rows: size,
            num_cols: size,
            data: vec![T::default(); (size * size) as usize].into_boxed_slice(),
            is_square: true,
        }
    }

    ///creates a identity matrix with the given size
    pub fn nml_mat_eye(size: u32) -> Self {
        let mut data: Vec<T> = vec![T::default(); (size * size) as usize];
        for i in 0..size {
            data[(i * size + i) as usize] = T::one();
        }
        Self {
            num_rows: size,
            num_cols: size,
            data: data.into_boxed_slice(),
            is_square: true,
        }
    }
    ///Creates a new matrix which is a copy of a given matrix. Uses only the reference to the matrix, so that the original matrix is not moved
    pub fn nml_mat_cp(matrix: &NmlMatrix<T>) -> Self {
        Self {
            num_rows: matrix.num_rows,
            num_cols: matrix.num_cols,
            data: matrix.data.clone(),
            is_square: matrix.is_square,
        }
    }

    ///Checks if two matrices are equal by checking their dimensions and after that every cell. Method is also used for implementation of trait PatialEq
    pub fn equality(self: &Self, matrix: &NmlMatrix<T>) -> bool {
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
    pub fn equality_in_tolerance(self: &Self, matrix: NmlMatrix<T>, tolerance: T) -> bool {
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

    ///Returns the value a specified at A[i,j]
    pub fn at(self: &Self, row: u32, col: u32) -> Result<T, NmlError> {
        match row < self.num_rows as u32 && col < self.num_cols as u32 {
            false => Err(NmlError::new(InvalidRows)),
            true => Ok(self.data[(row * self.num_cols + col) as usize]),
        }
    }

    ///Returns a result with a specified Column of a matrix, which in itself also is a matrix. If the specified matrix is not in the matrix the result will contain an error
    pub fn get_column(self: &Self, column: u32) -> Result<Self, NmlError> {
        match column < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                let mut data: Vec<T> = Vec::with_capacity(self.num_rows as usize);
                for i in 0..self.num_rows {
                    data.push(self.data[(i * self.num_rows + column) as usize]);
                }
                Ok(Self {
                    num_cols: 1,
                    num_rows: self.num_rows,
                    data: data.into_boxed_slice(),
                    is_square: false
                })
            },
        }
    }

    ///Returns a result which either contains a row of a matrix (which is also a matrix) or a error
    pub fn get_row(self: &Self, row: u32) -> Result<Self, NmlError> {
        match row < self.num_rows {
            true => {
                let data: Vec<T> = self.data[(row * self.num_cols) as usize..(row * self.num_cols + self.num_cols) as usize].to_vec().clone();
                Ok(Self {
                    num_cols: self.num_cols,
                    num_rows: 1,
                    data: data.into_boxed_slice(),
                    is_square: false
                })
            },
            false => Err(NmlError::new(InvalidRows)),
        }
    }
    /// Method sets the value of a given cell in the matrix through a mutable reference
    pub fn set_value(self: &mut Self, row: u32, col: u32, data: T) -> Result<(), NmlError> {
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
    pub fn set_all_values(self: &mut Self, value: T) {
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                self.data[(i * self.num_cols + j) as usize] = value;
            }
        }
    }
    ///checks if the matrix is square and sets the diagonal values to a given value
    pub fn set_dig_values(self: &mut Self, value: T) -> Result<(), NmlError> {
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
    pub fn multiply_row_scalar(self: &mut Self, row: u32, scalar: T) -> Result<(), NmlError> {
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
    pub fn multiply_col_scalar(self: &mut Self, col: u32, scalar: T) -> Result<(), NmlError> {
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
    pub fn multiply_matrix_scalar(self: &mut Self, scalar: T) {
        for i in 0..self.data.len() {
            self.data[i] *= scalar;
        }
    }

    /// row_1 is multiplied with scalar_1, this is analog for 2. row_1 will be modified with the solution (row_1 = row_1 * scalar + row_2 * scalar_2)
    pub fn add_rows(self: &mut Self, row_1: u32, scalar_1: T, row_2: u32, scalar_2: T) -> Result<(), NmlError> {
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
    pub fn swap_rows(self: &mut Self, row_1: u32, row_2: u32) -> Result<(), NmlError> {
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
    pub fn swap_columns(self: &mut Self, col_1: u32, col_2: u32) -> Result<(), NmlError> {
        match col_1 < self.num_cols && col_2 < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                for i in 0..self.num_rows {
                    let temp = self.data[(i * self.num_cols + col_1) as usize];
                    self.data[(i * self.num_cols + col_1) as usize] = self.data[(i * self.num_cols + col_2) as usize];
                    self.data[(i * self.num_cols + col_2) as usize] = temp;
                }
                Ok(())
            }
        }
    }
    ///Tries to remove a column of a matrix and returns the rest of the matrix as a now on. Does not move the original matrix.
    ///If the column is not in the original matrix the result will return an error
    pub fn remove_column(self: &Self, col: u32) -> Result<Self, NmlError> {
        match col < self.num_cols {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                let mut data: Vec<T> = Vec::with_capacity(self.data.len());
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
                        data: data.into_boxed_slice(),
                        is_square: false
                    }
                )
            }
        }
    }
    ///Tries to remove a column of a matrix and returns the rest of the matrix as a now on. Does not move the original matrix.
    ///If the column is not in the original matrix the result will return an error
    pub fn remove_row(self: &Self, row: u32) -> Result<Self, NmlError> {
        match row < self.num_rows {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                let data: Vec<T> = self.data[((row + 1) * self.num_cols) as usize..self.data.len()].to_vec();
                Ok(Self {
                    num_cols: self.num_cols,
                    num_rows: 1,
                    data: data.into_boxed_slice(),
                    is_square: false,
                })
            }
        }
    }

    pub fn get_sub_mtr(self: &Self, row_start: u32, row_end: u32, col_start: u32, col_end: u32) -> Result<Self, NmlError> {
        match row_start < self.num_rows && row_end < self.num_rows && col_start < self.num_cols && col_end < self.num_cols {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                let mut data: Vec<T> = Vec::new();
                for i in row_start - 1..row_end {
                    for j in col_start - 1..col_end {
                        data.push(self.data[(i * self.num_cols + j) as usize]);
                    }
                }
                Ok(Self {
                    num_rows: row_end - row_start,
                    num_cols: col_end - col_start,
                    data: data.into_boxed_slice(),
                    is_square: false,
                })
            }
        }
    }

    ///Computes the transpose matrix b' of a matrix b. This is achieved by going from row-major-ordering to a more efficient storage. The input matrix will not be modified or moved.
    pub fn transpose(self: &Self) -> Self{
        let mut data: Vec<T> = Vec::with_capacity(self.data.len());
        for i in 0..self.num_cols {
            for j in 0..self.num_rows {
                data.push(self.data[(i + j*self.num_cols) as usize]);
            }
        }
        Self {
            num_rows: self.num_cols,
            num_cols: self.num_rows,
            data: data.into_boxed_slice(),
            is_square: self.is_square,
        }
    }
    ///The naive matrix multiplication algorihm applied with the transponse of the one matrix
    pub fn mul_transpose(self: &Self, other: &Self) -> Result<Self, NmlError> {
        match self.num_cols == other.num_rows {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                let m: u32 = self.num_rows;
                let n: u32 = self.num_cols;
                let p: u32 = other.num_cols;
                let transpose: NmlMatrix<T> = other.transpose();
                let mut data: Vec<T> = Vec::new();
                for i in 0..m {
                    for j in 0..p {
                        data.insert((i * p + j) as usize, T::default());
                        for k in 0..n {
                            data[(i*p+j) as usize] += self.data[(i * n + k) as usize] * transpose.data[(p * k + j) as usize];
                        }
                    }
                }
                Ok(Self{
                    num_rows: self.num_rows,
                    num_cols: other.num_cols,
                    data: data.into_boxed_slice(),
                    is_square: self.num_rows == other.num_cols
                })
            }
        }
    }
    ///The naive matrix multiplication algorithm. It iterates trough all values of both matrices. These matrices are not moved or modified
    pub fn mul_naive(self: &Self, other: &Self) -> Result<Self,NmlError> {
        let m = self.num_rows;
        let n_1 = self.num_cols;
        let n_2 = other.num_rows;
        let p = other.num_cols;
        match n_1 == n_2 {
            false => {Err(NmlError::new(CreateMatrix))},
            true => {
                let mut data: Vec<T> = Vec::with_capacity((m*p) as usize);
                for i in 0..m {
                    for j in 0..p {
                        data.insert((i * p + j) as usize, T::default());
                        for k in 0..n_1 {
                            data[(i*p+j) as usize] += self.data[(i * n_1 + k) as usize] * other.data[(p * k + j) as usize];
                        }
                    }
                }
                Ok(Self{
                    num_rows: m,
                    num_cols: p,
                    data: data.into_boxed_slice(),
                    is_square: m == p,
                })
            }
        }
    }

    pub fn evcxr_display(&self) {
        let mut html = String::new();
        html.push_str("<table>");
        for i in 0..self.num_rows {
            html.push_str("<tr>");
            for j in 0..self.num_cols {
                html.push_str("<td>");
                html.push_str(&self.data[(i * self.num_cols + j) as usize].to_string());
                html.push_str("</td>");
            }
            html.push_str("</tr>");
        }
        html.push_str("</table>");
        println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", html);
    }

}
impl<T> Sub for NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign {
    type Output = Result<Self, NmlError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols {
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<T> = Vec::new();
                for i in 0..self.data.len() -1 {
                    data.push(self.data[i] - rhs.data[i]);
                }
                Ok(Self{
                    num_rows: self.num_rows,
                    num_cols: self.num_cols,
                    data: data.into_boxed_slice(),
                    is_square: self.is_square
                })
            }
        }
    }
}

impl<T> Sub for &NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign {
    type Output = Result<NmlMatrix<T>, NmlError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols {
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<T> = Vec::new();
                for i in 0..self.data.len() -1 {
                    data.push(self.data[i] - rhs.data[i]);
                }
                Ok(NmlMatrix{
                    num_rows: self.num_rows,
                    num_cols: self.num_cols,
                    data: data.into_boxed_slice(),
                    is_square: self.is_square
                })
            }
        }
    }
}

impl<T> Display for NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign + Display{
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

impl<T> Eq for NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign + SampleUniform + Display{}
impl<T> PartialEq for NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign + SampleUniform + Display{
    fn eq(&self, other: &Self) -> bool {
        self.equality(other)
    }
}

impl<T> Add for NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign{
    type Output = Result<Self, NmlError>;

    fn add(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols{
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<T> = Vec::new();
                for i in 0..self.data.len() {
                    data.push(self.data[i] + rhs.data[i]);
                }
                Ok(Self{
                    num_cols: self.num_cols,
                    num_rows: self.num_rows,
                    data: data.into_boxed_slice(),
                    is_square: self.is_square
                })
            }
        }
    }
}

impl<T> Add for &NmlMatrix<T>  where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign{
    type Output = Result<NmlMatrix<T>, NmlError>;

    fn add(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols{
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<T> = Vec::new();
                for i in 0..self.data.len() {
                    data.push(self.data[i] + rhs.data[i]);
                }
                Ok(NmlMatrix{
                    num_cols: self.num_cols,
                    num_rows: self.num_rows,
                    data: data.into_boxed_slice(),
                    is_square: self.is_square
                })
            }
        }
    }
}

impl<T> Mul for NmlMatrix<T> where T: Num + Copy + Default + Signed + PartialOrd + MulAssign + AddAssign + SampleUniform + Display{
    type Output = Result<Self<>, NmlError>;

    fn mul(self, rhs: Self) -> Self::Output {
        return self.mul_naive(&rhs);
    }
}

