package main

import (
	"flag"
	"io/ioutil"
	"log"
	"path/filepath"
	"sort"

	"github.com/disintegration/imaging"
	"github.com/k0kubun/pp"
	utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// Parse flags
	modeldir :