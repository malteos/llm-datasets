install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ".[all]"

install-for-tests:
	@echo "--- 🚀 Installing project dependencies for test ---"
	@echo "This ensures that the project is not installed in editable mode"
	pip install ".[dev]"

install-tlsh:
	@echo "--- 🚀 Installing TLSH dependency (same version as OSCAR 23.01) ---"
	pip download python-tlsh==4.5.0 && \
		tar -xvf python-tlsh-4.5.0.tar.gz && \
		cd python-tlsh-4.5.0 && \
		sed -i 's/set(TLSH_BUCKETS_128 1)/set(TLSH_BUCKETS_256 1)/g; s/set(TLSH_CHECKSUM_1B 1)/set(TLSH_CHECKSUM_3B 1)/g' CMakeLists.txt && \
		python setup.py install && \
		rm -rf python-tlsh-4.5.0*

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 			# running ruff formatting
	ruff check . --fix  	# running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	# Required for CI to work, otherwise it will just pass
	ruff format . --check						    # running ruff formatting
	ruff check **/*.py 						        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest --durations=5 ./tests

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test
