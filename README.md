# The New-Matrix-Library (NML)   
## While writitng this library i followed the tutorial from [Andrei Ciobanu](https://www.andreinc.net/2021/01/20/writing-your-own-linear-algebra-matrix-library-in-c)

This library is a personal project of mine to learn more about linear algebra and get more into rust-programming.
You can also find the Library on [Crates.io](https://crates.io/crates/nml-matrix) and add it to your projects with the cargo package manager.

//write a good readme for the Library which includes the link to docs.rs the Struct of the Matrix is called NmlMatrix

//write a good readme for the Library which includes the link to docs.rs the Struct of the Matrix is called NmlMatrix

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
nml-matrix = "0.2.1"
```

## Example
```rust
use nml_matrix::NmlMatrix;

fn main() {
    let a = NmlMatrix::new_with_data(3, 2, vec![1, 2, 3, 4, 5, 6]);
    let b = NmlMatrix::new_with_data(3, 2, vec![1, 2, 3, 4, 5, 6]);
    let c = a * b;
    println!("{:?}", c);
}
```

## Documentation
You can find the documentation [here](https://docs.rs/nml-matrix/0.1.0/nml_matrix/struct.NmlMatrix.html)

## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE-File](LICENSE) file for details