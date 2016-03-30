package main

import (
	"bytes"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", mainHandler)
	http.ListenAndServe("0.0.0.0:8000", mux)
}

func mainHandler(w http.ResponseWriter, r *http.Request) {
	var buf bytes.Buffer
	buf.ReadFrom(r.Body)

	inputstr := strings.Split(buf.String(), ",")

	if len(inputstr) != 1024 {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	inputs := make([]float32, 1024)

	for i, str := range inputstr {
		f, err := strconv.ParseFloat(str, 32)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		inputs[i] = float32(f)
	}

	ctx := createNaclContext()

	start := time.Now()

	out := ctx.run(inputs)

	fmt.Printf("Run time: %s\n", time.Since(start))

	outStr := make([]string, 8)

	for i, f := range out {
		outStr[i] = strconv.FormatFloat(float64(f), 'E', -1, 32)
	}

	fmt.Fprint(w, strings.Join(outStr, ","))
}
