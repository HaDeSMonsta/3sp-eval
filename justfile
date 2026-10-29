default:
	@just --list

run:
	nix-shell --run ".venv/bin/python main.py"

run-full:
	rm -r data-out/ || continue
	nix-shell --run ".venv/bin/python main.py"
