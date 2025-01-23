let
  pkgs = import <nixpkgs> {};
  python = pkgs.python311;
  pythonPackages = python.pkgs;
in

with pkgs;

mkShell {
  name = "pip-env";
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.zlib
    pkgs.libGL
    pkgs.glib
    pkgs.stdenv.cc.cc
  ];
  buildInputs = with pythonPackages; [
    pkgs.poetry
    ipykernel
    jupyter-all
    # pytest
    setuptools
    wheel
    pypy310
  ];
}
