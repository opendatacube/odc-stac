DKR ?= kirillodc/odc-stac-binder:latest
ALL_PY := $(wildcard *.py)
ALL_NB := $(patsubst %.py, %.ipynb, $(ALL_PY))

all: $(ALL_NB)

%.ipynb: %.py
	@echo $< "=>" $@
	docker run -i --entrypoint ./binder/render-nb-pipe.sh $(DKR) < $< > $@

debug:
	docker run --rm -ti --entrypoint /bin/bash -v $$(pwd):/home/jovyan/notebooks $(DKR)

clean:
	@echo Removing Rendered Notebooks
	rm -f $(ALL_NB)

.PHONY: debug all clean
