
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	//Parse flags
	modeldir := flag.String("dir", "./", "Directory containing trained model files. Assumes model file is called frozen_inference_graph.pb")
	//datafile := flag.String()
	flag.Parse()
	if *modeldir == "" {
		flag.Usage()
		return
	}

	modelpath := filepath.Join(*modeldir, "DIEN.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}
	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	source := Dataprocess(16)
	uids, mids, cats, mid_his, cat_his, mid_mask, sl := prepare_data(source)

	// uids, mids, cats, mid_his, cat_his, mid_mask, length_x
	inputOp0 := graph.Operation("Inputs/mid_his_batch_ph")
	inputOp1 := graph.Operation("Inputs/cat_his_batch_ph")
	inputOp2 := graph.Operation("Inputs/uid_batch_ph")
	inputOp3 := graph.Operation("Inputs/mid_batch_ph")
	inputOp4 := graph.Operation("Inputs/cat_batch_ph")
	inputOp5 := graph.Operation("Inputs/mask")
	inputOp6 := graph.Operation("Inputs/seq_len_ph")

	// inputOp7 := graph.Operation("Inputs/noclk_mid_batch_ph")
	// inputOp8 := graph.Operation("Inputs/noclk_cat_batch_ph")
	// inputOp9 := graph.Operation("Inputs/target_ph")

	o1 := graph.Operation("dien/fcn/Softmax")
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp0.Output(0): mid_his,
			inputOp1.Output(0): cat_his,
			inputOp2.Output(0): uids,
			inputOp3.Output(0): mids,
			inputOp4.Output(0): cats,
			inputOp5.Output(0): mid_mask,
			inputOp6.Output(0): sl,
			// inputOp7.Output(0): noClkMidHis,
			// inputOp8.Output(0): noClkCatHis,
			// inputOp9.Output(0): target,
		},
		[]tf.Output{
			o1.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	probabilities := output[0].Value().([][]float32)[0]
	fmt.Println(probabilities)

}

func prepare_data(source [][]interface{}) (*tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor, *tf.Tensor) {
	lengthx := []int32{}
	seqs_mid := [][]int{}
	seqs_cat := [][]int{}

	uidsRaw := []int32{}
	midsRaw := []int32{}
	catsRaw := []int32{}

	for _, v := range source {
		uidsRaw = append(uidsRaw, int32(v[0].(int)))
		midsRaw = append(midsRaw, int32(v[1].(int)))
		catsRaw = append(catsRaw, int32(v[2].(int)))
		lengthx = append(lengthx, int32(len(v[4].([]int))))
		seqs_mid = append(seqs_mid, v[3].([]int))
		seqs_cat = append(seqs_cat, v[4].([]int))
	}

	uids, err := tf.NewTensor(uidsRaw)
	if err != nil {
		log.Fatal(err)

	}
	mids, _ := tf.NewTensor(midsRaw)
	cats, _ := tf.NewTensor(catsRaw)

	n_samples := len(seqs_mid)
	var maxlen_x int32
	for i, e := range lengthx {
		if i == 0 || e >= maxlen_x {
			maxlen_x = e
		}
	}

	mid_his_raw := make([][]int32, n_samples)
	for n := 0; n < n_samples; n++ {
		tn := make([]int32, maxlen_x)
		for m := 0; m < int(maxlen_x); m++ {
			tn[m] = 0
		}
		mid_his_raw[n] = tn
	}

	cat_his_raw := make([][]int32, n_samples)