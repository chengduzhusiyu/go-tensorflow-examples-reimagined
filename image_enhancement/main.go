package main

import (
	"bytes"
	"flag"
	"image"
	"image/color"
	"image/png"
	_ "image/png"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/k0kubun/pp"

	"github.com/disintegration/imaging"
	utils "github.com/rai