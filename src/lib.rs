mod matrix;
mod util;
use crate::matrix::NmlMatrix;

///unit tests for the Methods and Constructors of the NmlMatrix struct
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nml_identity_and_matrix_equality() {
        let data: Vec<f64> = vec![1_f64,0_f64,0_f64,1_f64];
        let matrix_1 = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let matrix_2 = NmlMatrix::nml_mat_eye(2);
        let result: bool = matrix_1== matrix_2;
        assert_eq!(result, true);
    }

    #[test]
    fn nml_identity_and_matrix_equality_in_tolerance() {
        let data: Vec<f64> = vec![2_f64,1_f64,1_f64,2_f64];
        let matrix_1 = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let matrix_2 = NmlMatrix::nml_mat_eye(2);
        let result: bool = matrix_1.equality_in_tolerance(matrix_2, 1_f64);
        assert_eq!(result, true);
    }

    #[test]
    fn nml_identity_and_matrix_inequality() {
        let data: Vec<f64> = vec![1_f64,0_f64,1_f64,1_f64];
        let matrix_1 = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let matrix_2 = NmlMatrix::nml_mat_eye(2);
        let result: bool = matrix_1 == matrix_2;
        assert_eq!(result, false);
    }

    #[test]
    fn get_column_equal() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let column = matrix.get_column(1).expect("Unable to get column");
        let expected_data: Vec<f64> = vec![2_f64,4_f64];
        let expected = NmlMatrix::new_with_data(2, 1, expected_data).expect("Unable to create matrix");
        assert_eq!(column == expected, true);
    }

    #[test]
    fn get_column_unequal() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let column = matrix.get_column(1).expect("Unable to get column");
        let expected_data: Vec<f64> = vec![1_f64,4_f64];
        let expected = NmlMatrix::new_with_data(2, 1, expected_data).expect("Unable to create matrix");
        assert_eq!(column == expected, false);
    }

    #[test]
    fn copy_matrix_equality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let copy = NmlMatrix::nml_mat_cp(&matrix);
        assert_eq!(matrix == copy, true);
    }

    #[test]
    fn copy_matrix_inequality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let copy = NmlMatrix::nml_mat_cp(&matrix);
        let mut copy_data = copy.data;
        copy_data[0] = 0_f64;
        let copy = NmlMatrix::new_with_data(2, 2, copy_data).expect("Unable to create matrix");
        assert_eq!(matrix == copy, false);
    }

    #[test]
    fn rnd_matrix_tolerance_equality() {
        let rnd_matrix: NmlMatrix = NmlMatrix::nml_mat_rnd(2, 2, 0_f64, 1_f64);
        let matrix: NmlMatrix = NmlMatrix::nml_mat_sqr(2);
        assert_eq!(rnd_matrix.equality_in_tolerance(matrix, 1_f64), true);
    }

    #[test]
    fn rnd_matrix_tolerance_inequality() {
        let rnd_matrix: NmlMatrix = NmlMatrix::nml_mat_rnd(2, 2, 2_f64, 3_f64);
        let matrix: NmlMatrix = NmlMatrix::nml_mat_sqr(2);
        assert_eq!(rnd_matrix.equality_in_tolerance(matrix, 1_f64), false);
    }

    #[test]
    pub fn get_column_invalid() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let column = matrix.get_column(2);
        assert_eq!(column.is_err(), true);
    }

    #[test]
    pub fn get_row_equality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let matrix = NmlMatrix::new_with_data(1, 4, data).expect("Unable to create matrix");
        let row = matrix.get_row(0).expect("Unable to get row");
        assert_eq!(matrix == row, true);
    }

    #[test]
    pub fn get_row_inequality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let matrix = NmlMatrix::new_with_data(1, 4, data).expect("Unable to create matrix");
        let row = matrix.get_row(0).expect("Unable to get row");
        let mut row_data = row.data;
        row_data[0] = 0_f64;
        let row = NmlMatrix::new_with_data(1, 4, row_data).expect("Unable to create matrix");
        assert_eq!(matrix == row, false);
    }

    #[test]
    pub fn set_all_values_equality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(1, 4, data).expect("Unable to create matrix");
        matrix.set_all_values(0_f64);
        let expected_data: Vec<f64> = vec![0_f64,0_f64,0_f64, 0_f64];
        let expected = NmlMatrix::new_with_data(1, 4, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn set_all_values_ineqality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(1, 4, data).expect("Unable to create matrix");
        matrix.set_all_values(0_f64);
        let expected_data: Vec<f64> = vec![0_f64,0_f64,0_f64, 1_f64];
        let expected = NmlMatrix::new_with_data(1, 4, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, false);
    }

    #[test]
    pub fn set_value_equality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(1, 4, data).expect("Unable to create matrix");
        matrix.set_value(0, 0, 0_f64);
        let expected_data: Vec<f64> = vec![0_f64,2_f64,3_f64, 4_f64];
        let expected = NmlMatrix::new_with_data(1, 4, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn set_value_inqality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(1, 4, data).expect("Unable to create matrix");
        let expected = NmlMatrix::nml_mat_cp(&matrix);
        matrix.set_value(0, 0, 0_f64);
        assert_eq!(matrix == expected, false);
    }

    #[test]
    pub fn set_diagonal_values_equality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        matrix.set_dig_values(0_f64).expect("Matrix is not square");
        let expected_data: Vec<f64> = vec![0_f64,2_f64,3_f64, 0_f64];
        let expected = NmlMatrix::new_with_data(2, 2, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn  multiply_column_scalar() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        matrix.multiply_col_scalar(0, 2_f64).expect("Invalid column");
        let expected_data: Vec<f64> = vec![2_f64,2_f64,6_f64, 4_f64];
        let expected = NmlMatrix::new_with_data(2, 2, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn multiply_row_scalar() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        matrix.multiply_row_scalar(0, 2_f64).expect("Invalid row");
        let expected_data: Vec<f64> = vec![2_f64,4_f64,3_f64, 4_f64];
        let expected = NmlMatrix::new_with_data(2, 2, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn multiply_matrix_scalar() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        matrix.multiply_matrix_scalar(2_f64);
        let expected_data: Vec<f64> = vec![2_f64,4_f64,6_f64, 8_f64];
        let expected = NmlMatrix::new_with_data(2, 2, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn add_rows() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64];
        let mut matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        matrix.add_rows(0, 2_f64, 1, 1_f64).expect("Invalid row");
        let expected_data: Vec<f64> = vec![5_f64,8_f64,3_f64, 4_f64];
        let expected = NmlMatrix::new_with_data(2, 2, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }

    #[test]
    pub fn remove_column() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64, 5_f64, 6_f64];
        let matrix = NmlMatrix::new_with_data(3, 2, data).expect("Unable to create matrix");
        let result = matrix.remove_column(0).expect("Invalid column");
        let expected_data: Vec<f64> = vec![2_f64,4_f64, 6_f64];
        let expected = NmlMatrix::new_with_data(3, 1, expected_data).expect("Unable to create matrix");
        assert_eq!(result == expected, true);
    }

    #[test]
    pub fn remove_row() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64, 5_f64, 6_f64];
        let matrix = NmlMatrix::new_with_data(2, 3, data).expect("Unable to create matrix");
        let result = matrix.remove_row(0).expect("Invalid row");
        let expected_data: Vec<f64> = vec![4_f64,5_f64, 6_f64];
        let expected = NmlMatrix::new_with_data(1, 3, expected_data).expect("Unable to create matrix");
        assert_eq!(result == expected, true);
    }

    #[test]
    pub fn swap_row() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64, 4_f64, 5_f64, 6_f64];
        let mut matrix = NmlMatrix::new_with_data(2, 3, data).expect("Unable to create matrix");
        matrix.swap_rows(0, 1).expect("Invalid row");
        let expected_data: Vec<f64> = vec![4_f64,5_f64, 6_f64, 1_f64,2_f64,3_f64];
        let expected = NmlMatrix::new_with_data(2, 3, expected_data).expect("Unable to create matrix");
        assert_eq!(matrix == expected, true);
    }
}
