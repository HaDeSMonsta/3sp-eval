{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
let
  libs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    bzip2
    libffi
    openssl
    xz
    libxml2
    sqlite
    curl
    expat
    libpng
    freetype
    libtiff
    libjpeg
    krb5
    libuuid
    ncurses
    readline
    libiconv
  ];
in
pkgs.mkShell {
  packages = libs ++ [ pkgs.corefonts ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export QUADTREE_FONT_DIR="${pkgs.corefonts}/share/fonts/truetype"
  '';
}
