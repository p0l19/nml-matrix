mod matrix;
mod util;
use crate::matrix::NmlMatrix;
use crate::util::{NmlError, ErrorKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nml_matrix_constuctor() {
        let data: Vec<f64> = vec![1_f64,0,0,1];
        let matrix = NmlMatrix{
            num_rows: 2,
            num_cols: 2,
            data: data,
            is_square: true,
        };
        assert_eq!(result, 4);
    }
}
