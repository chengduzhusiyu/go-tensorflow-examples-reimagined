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
	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func drawImagefromArray(input [][][]float32, fileName string, width, height int) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	var R, G, B uint8
	for w := 0; w < width; w++ {
		for h := 0; h < height; h++ {
			R, G, B = uint8((input[h][w][0]+1)*127.5), uint8((input[h][w][1]+1)*127.5), uint8((input[h][w][2]+1)*127.5)
			img.Set(w, h, color.RGBA{R, G, B, 255})
		}
	}
	pp.Println(img.At(0, 0))

	// Save to output.png
	out, _ := os.Create(fileName)
	defer out.Close()

	err := png.Encode(out, img)
	if err != nil {
		log.Println(err)
	}
}

