MSG = "Build at $(shell /bin/date '+%Y-%m-%d %H-%M-%S')"
upload: 
	rm -rf ./docs && hugo
	git add -A
	git commit -m $(MSG)
	git push origin master