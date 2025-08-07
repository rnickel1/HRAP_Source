## User Installation Instructions (Windows/Linux/Mac)
First install Python https://www.python.org/downloads/. Version >=3.10 is required.

Then open your terminal and execute:

```
python3 -m pip install --upgrade pip

python3 -m pip install hrap
```

To update hrap when new features become available, simply run

```
python3 -m pip install --upgrade hrap
```

To uninstall hrap,

```
python3 -m pip uninstall hrap
```

## Running HRAP
To start the GUI, simply run the command from any directory:

```
hrap
```

If you are interested in advanced usage such as design optimization, uncertainty quantification, etc. then scripting with the Python interface is recommended.
See the "HRAP - Python/hrap/examples" directories for a few examples.

Coming soon: optimization and UQ examples

## Advanced Usage
HRAP Python utilizes JAX for Just-In-Time (JIT) compilation, enabling much faster execution speeds that make the real-time GUI updates possible and facilitate speedy parameter studies.
The helper functions in HRAP minimize the technical knowledge of JAX and Python required to make modifications.
Still, a few 

Coming soon: example on how to add a custom ???

## Developer Installation Instructions (Windows/Linux/Mac)
This is only necessary if you wish to contribute to the official HRAP repository.
You can script your own custom functionality with the "user installation."
This is an alternative to "User Installation." Please uninstall hrap or use a new venv before switching to the "Developer Installation."

After cloning the repo,

```
cd "HRAP_JAX/HRAP - Python/"

pythom3 -m pip install -e ./
```

You can now run the GUI as usual. Your local modifications to the hrap code will be reflected.
