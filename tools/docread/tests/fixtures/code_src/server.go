package main

import (
	"fmt"
	"net/http"
)

type Handler struct {
	Name string
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "hello from %s", h.Name)
}

func main() {
	http.Handle("/", &Handler{Name: "garden"})
	http.ListenAndServe(":8080", nil)
}
