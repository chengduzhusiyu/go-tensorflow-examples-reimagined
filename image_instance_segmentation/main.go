package main

import (
	"flag"
	"image"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tenso