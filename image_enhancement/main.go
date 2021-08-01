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

func makeTensorFromImage(filename string) (*tf.Tensor, image.Image, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal(err)
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)

	if err != nil {
		log.Fatal(err)
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		log.Fatal(err)
	}

	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := constructGraphToNormalizeImage()
	if err != nil {
		log.Fatal(err)
	}

	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	return normalized[0], img, nil
}

func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		Scale = float32(127.5)
		Mean  = float32(1)
	)

	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output =
		op.Sub(s,
			op.Div(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodePng(s, input, op.DecodePngChannels(3)),
						tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("scale"), Scale)),
			op.Const(s.SubScope("mean"), Mean))

	graph, err = s.Finalize()
	return graph, input, output, err
}

func main() {
	// Parse flags
	modelDir := flag.String("dir", ".", "Directory containing trained model files")
	pngFile := flag.String("png", "penguin.png", "Path of a PNG