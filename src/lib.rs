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
        let result: bool = matrix_1.equality(matrix_2);
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
        let result: bool = matrix_1.equality(matrix_2);
        assert_eq!(result, false);
    }

    #[test]
    fn get_column_equal() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let column = matrix.get_column(1).expect("Unable to get column");
        let expected_data: Vec<f64> = vec![2_f64,4_f64];
        let expected = NmlMatrix::new_with_data(2, 1, expected_data).expect("Unable to create matrix");
        assert_eq!(column.equality(expected), true);
    }

    #[test]
    fn get_column_unequal() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let column = matrix.get_column(1).expect("Unable to get column");
        let expected_data: Vec<f64> = vec![1_f64,4_f64];
        let expected = NmlMatrix::new_with_data(2, 1, expected_data).expect("Unable to create matrix");
        assert_eq!(column.equality(expected), false);
    }

    #[test]
    fn copy_matrix_equality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let copy = NmlMatrix::nml_mat_cp(&matrix);
        assert_eq!(matrix.equality(copy), true);
    }

    #[test]
    fn copy_matrix_inequality() {
        let data: Vec<f64> = vec![1_f64,2_f64,3_f64,4_f64];
        let matrix = NmlMatrix::new_with_data(2, 2, data).expect("Unable to create matrix");
        let copy = NmlMatrix::nml_mat_cp(&matrix);
        let mut copy_data = copy.data;
        copy_data[0] = 0_f64;
        let copy = NmlMatrix::new_with_data(2, 2, copy_data).expect("Unable to create matrix");
        assert_eq!(matrix.equality(copy), false);
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
        assert_eq!(matrix.equality(row), true);
    }
}
