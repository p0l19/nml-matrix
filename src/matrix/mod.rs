use std::fmt::Display;
use std::ops::{Add, Mul, Sub};
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

    pub fn new_with_2d_vec(num_rows: u32, num_cols: u32, data: Vec<Vec<f64>>) -> Result<Self, NmlError> {
        todo!("Not implemented yet");
    }

    ///Constructor that uses a vector to initialize the matrix. checks if the entered rows and columns fit the vector size
    pub fn new_with_data(num_rows: u32, num_cols: u32, data: Vec<f64>) -> Result<Self, NmlError> {
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

    pub fn get_value(self: &Self, row: i32, col: i32) -> Result<f64, NmlError> {
        match row < self.num_rows as i32 && col < self.num_cols as i32 {
            false => Err(NmlError::new(InvalidRows)),
            true => Ok(self.data[(row * self.num_cols as i32 + col) as usize]),
        }
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
    pub fn multiply_col_scalar(self: &mut Self, col: u32, scalar: f64) -> Result<(), NmlError> {
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
    pub fn multiply_matrix_scalar(self: &mut Self, scalar: f64) -> Self {
        let mut data: Vec<f64> = Vec::new();
        for i in 0..self.data.len() {
            data.push(self.data[i] * scalar);
        }
        Self {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            data,
            is_square: self.is_square,
        }
    }

    /// row_1 ist multiplied with scalar_1, this is analog for 2. row_1 will be modified with the solution (row_1 = row_1 * scalar + row_2 * scalar_2)
    pub fn add_rows(self: &mut Self, row_1: u32, scalar_1: f64, row_2: u32, scalar_2: f64) -> Result<(), NmlError> {
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
    pub fn remove_row(self: &Self, row: u32) -> Result<Self, NmlError> {
        match row < self.num_rows {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                let data: Vec<f64> = self.data[((row + 1) * self.num_cols) as usize..self.data.len()].to_vec();
                Ok(Self {
                    num_cols: self.num_cols,
                    num_rows: 1,
                    data,
                    is_square: false,
                })
            }
        }
    }

    pub fn get_sub_mtr(self: &Self, row_start: u32, row_end: u32, col_start: u32, col_end: u32) -> Result<Self, NmlError> {
        match row_start < self.num_rows && row_end < self.num_rows && col_start < self.num_cols && col_end < self.num_cols {
            false => Err(NmlError::new(InvalidRows)),
            true => {
                let mut data: Vec<f64> = Vec::new();
                for i in row_start - 1..row_end {
                    for j in col_start - 1..col_end {
                        data.push(self.data[(i * self.num_cols + j) as usize]);
                    }
                }
                Ok(Self {
                    num_rows: row_end - row_start,
                    num_cols: col_end - col_start,
                    data,
                    is_square: false,
                })
            }
        }
    }

    ///Computes the transpose matrix b' of a matrix b. This is achieved by going from row-major-ordering to a more efficient storage. The input matrix will not be modified or moved.
    pub fn transpose(self: &Self) -> Self{
        let mut data: Vec<f64> = Vec::with_capacity(self.data.len());
        for i in 0..self.num_cols {
            for j in 0..self.num_rows {
                data.push(self.data[(i + j*self.num_cols) as usize]);
            }
        }
        Self {
            num_rows: self.num_cols,
            num_cols: self.num_rows,
            data,
            is_square: self.is_square,
        }
    }

    pub fn mul_transpose(self: &Self, other: &Self) -> Result<Self, NmlError> {
        match self.num_cols == other.num_rows {
            false => Err(NmlError::new(InvalidCols)),
            true => {
                let m: u32 = self.num_rows;
                let n: u32 = self.num_cols;
                let transpose: NmlMatrix = other.transpose();
                let mut data: Vec<f64> = Vec::new();

                Ok(Self{
                    num_rows: self.num_rows,
                    num_cols: other.num_cols,
                    data,
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
                let mut data = Vec::with_capacity((m*p) as usize);
                for i in 0..m {
                    for j in 0..p {
                        data.insert((i * p + j) as usize, 0.0);
                        for k in 0..n_1 {
                            data[(i*p+j) as usize] += self.data[(i * n_1 + k) as usize] * other.data[(p * k + j) as usize];
                        }
                    }
                }
                Ok(Self{
                    num_rows: m,
                    num_cols: p,
                    data,
                    is_square: m == p,
                })
            }
        }
    }


    /// The method expects that only two square matrices of the same size are entered. Where n is a power of 2. Therefore the method is private and should only be called from the mul-trait
    fn strassen_algorithm(matrix_1: &NmlMatrix, matrix_2: &NmlMatrix) -> Self {
        let dimensions: u32 = matrix_1.num_rows;
        if dimensions <= 2 {
            return matrix_1.mul_naive(&matrix_2).expect("Matrix size does not match");
        }
        else {
            //get the four sub matrices of the original matrix that constitute each quadrant.
            let a_11: NmlMatrix = matrix_1.get_sub_mtr(1, dimensions / 2, 1, dimensions / 2).expect("Submatrix could not be created");
            let a_12: NmlMatrix = matrix_1.get_sub_mtr(1, dimensions / 2, dimensions / 2, dimensions-1).expect("Submatrix could not be created");
            let a_21: NmlMatrix = matrix_1.get_sub_mtr(dimensions / 2, dimensions-1, 1, dimensions / 2).expect("Submatrix could not be created");
            let a_22: NmlMatrix = matrix_1.get_sub_mtr(dimensions / 2, dimensions-1, dimensions / 2, dimensions-1).expect("Submatrix could not be created");

            let b_11: NmlMatrix = matrix_2.get_sub_mtr(1, dimensions / 2, 1, dimensions / 2).expect("Submatrix could not be created");
            let b_12: NmlMatrix = matrix_2.get_sub_mtr(1, dimensions / 2, dimensions / 2, dimensions-1).expect("Submatrix could not be created");
            let b_21: NmlMatrix = matrix_2.get_sub_mtr(dimensions / 2, dimensions-1, 1, dimensions / 2).expect("Submatrix could not be created");
            let b_22: NmlMatrix = matrix_2.get_sub_mtr(dimensions / 2, dimensions-1, dimensions / 2, dimensions-1).expect("Submatrix could not be created");
            //Compute the intermediate matrices m1 - m7. Uses only 7 matrix multipliations
            let m_1: NmlMatrix = Self::strassen_algorithm(&(&a_11 + &a_22).expect(""), &(&b_11 + &b_22).expect(""));
            let m_2: NmlMatrix = Self::strassen_algorithm(&(&a_21 + &a_22).expect(""), &b_11);
            let m_3: NmlMatrix = Self::strassen_algorithm(&a_11, &(&b_12 - &b_22).expect(""));
            let m_4: NmlMatrix = Self::strassen_algorithm(&a_22, &(&b_21 - &b_11).expect(""));
            let m_5: NmlMatrix = Self::strassen_algorithm(&(&a_11 + &a_12).expect(""), &b_22);
            let m_6: NmlMatrix = Self::strassen_algorithm(&(&a_21 - &a_11).expect(""), &(&b_11 + &b_12).expect(""));
            let m_7: NmlMatrix = Self::strassen_algorithm(&(&a_12 - &a_22).expect(""), &(&b_21 + &b_22).expect(""));
            //Add the intermediate matrices to get the sub-matrices of the result c
            let c_11: NmlMatrix = ((&m_1 + &m_4).expect("") - (&m_5 + &m_7).expect("")).expect("");
            let c_12: NmlMatrix = (&m_3 + &m_5).expect("");
            let c_21: NmlMatrix = (&m_2 + &m_4).expect("");
            let c_22: NmlMatrix = ((&m_1 - &m_2).expect("") + (m_3 + m_6).expect("")).expect("");
            //reconstitute the sub-matrices from c into c
            let mut data: Vec<f64> = Vec::new();
            data.append(&mut c_11.data[0usize..(dimensions / 2) as usize].to_vec());
            data.append(&mut c_12.data[0usize..(dimensions / 2) as usize].to_vec());
            data.append(&mut c_11.data[(dimensions / 2) as usize ..dimensions as usize].to_vec());
            data.append(&mut c_12.data[(dimensions / 2) as usize ..dimensions as usize].to_vec());
            data.append(&mut c_21.data[0usize..(dimensions / 2) as usize].to_vec());
            data.append(&mut c_22.data[0usize..(dimensions / 2) as usize].to_vec());
            data.append(&mut c_21.data[(dimensions / 2) as usize ..dimensions as usize].to_vec());
            data.append(&mut c_22.data[(dimensions / 2) as usize ..dimensions as usize].to_vec());

            Self {
                num_rows: dimensions,
                num_cols: dimensions,
                data,
                is_square: true,
            }
        }
    }
    ///The purpose of this function is to prepare the matrices to be used in the strassen algorithm. This means they need to become square, where the dimension is a power of 2.
    ///The pre_strassen function assumes that matrix_1 and matrix_2 are such, that there exists a matrix C = matrix_1 * matrix_2. Therefore it does not again check if matrix_1.num_rows == matrix_2.num_cols
    fn pre_strassen(matrix_1: &NmlMatrix, matrix_2: &NmlMatrix) -> (NmlMatrix, NmlMatrix) {
        let mut biggest_dimension: u32 = matrix_1.num_rows;
        if (matrix_1.num_cols > biggest_dimension) && (matrix_1.num_cols > matrix_2.num_rows) {
            biggest_dimension = matrix_1.num_cols;
        }
        else if matrix_2.num_rows > biggest_dimension {
            biggest_dimension = matrix_2.num_rows;
        }

        if biggest_dimension%2 == 1 {
            biggest_dimension+=1;
        }

        return if matrix_1.is_square && matrix_2.is_square && matrix_1.num_rows == biggest_dimension {
            (NmlMatrix::nml_mat_cp(matrix_1), NmlMatrix::nml_mat_cp(matrix_2))
        } else {
            (NmlMatrix::nml_mat_cp(matrix_1).pad(biggest_dimension), NmlMatrix::nml_mat_cp(matrix_2).pad(biggest_dimension))
        }
    }

    pub fn pad(self: &Self, dimension: u32) -> Self {
        if self.is_square {
            //Find the difference between dimension and the size of self. Add as many rows and columns of zeros.
            let difference: u32 = dimension - self.num_cols;
            if difference == 0 {
                return NmlMatrix::nml_mat_cp(self)
            }
            let mut data: Vec<f64> = self.data.clone();
            for i in 0..difference {

            }
            Self {
                num_rows: dimension,,
                num_cols
                num_cols: dimension,
                data,
                is_square: true
            }
        }
        else {
            //Add rows and columns of 0 so that Self is square with a size of dimension
            let mut data: Vec<f64> = Vec::new();
            Self {
                num_rows: dimension,
                num_cols: dimension,
                data,
                is_square: true
            }
        }
    }

    pub fn reduce(self: &mut Self) -> Self{

    }
}
impl Sub for NmlMatrix{
    type Output = Result<Self, NmlError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols {
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<f64> = Vec::new();
                for i in 0..self.data.len() -1 {
                    data.push(self.data[i] - rhs.data[i]);
                }
                Ok(Self{
                    num_rows: self.num_rows,
                    num_cols: self.num_cols,
                    data,
                    is_square: self.is_square
                })
            }
        }
    }
}

impl Sub for &NmlMatrix {
    type Output = Result<NmlMatrix, NmlError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols {
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<f64> = Vec::new();
                for i in 0..self.data.len() -1 {
                    data.push(self.data[i] - rhs.data[i]);
                }
                Ok(NmlMatrix{
                    num_rows: self.num_rows,
                    num_cols: self.num_cols,
                    data,
                    is_square: self.is_square
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
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols{
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<f64> = Vec::new();
                for i in 0..self.data.len() {
                    data.push(self.data[i] + rhs.data[i]);
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

impl Add for &NmlMatrix {
    type Output = Result<NmlMatrix, NmlError>;

    fn add(self, rhs: Self) -> Self::Output {
        match self.num_rows == rhs.num_rows && self.num_cols == rhs.num_cols{
            false => Err(NmlError::new(CreateMatrix)),
            true => {
                let mut data: Vec<f64> = Vec::new();
                for i in 0..self.data.len() {
                    data.push(self.data[i] + rhs.data[i]);
                }
                Ok(NmlMatrix{
                    num_cols: self.num_cols,
                    num_rows: self.num_rows,
                    data,
                    is_square: self.is_square
                })
            }
        }
    }
}

impl Mul for NmlMatrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let matricies: (NmlMatrix, NmlMatrix) = NmlMatrix::pre_strassen(&self, &rhs);
        let result: NmlMatrix = NmlMatrix::strassen_algorithm(&matricies.0, &matricies.1);
        return result.reduce();


    }
}

