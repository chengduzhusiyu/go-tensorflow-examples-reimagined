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
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"golang.org/x/image/colornames"
)

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing tra