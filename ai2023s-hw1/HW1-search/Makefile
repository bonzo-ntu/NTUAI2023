# color
BLUE = \033[34m
NC = \033[0m

autograder: ## make autograder q=[q1, q2, q3, q4]
	python autograder.py -q $(q)

pacman: ## pacman maze=[tinyMaze, mediumMaze, bigMaze] fn=[tinyMazeSearch, dfs, bfs, ucs, astar]
	python pacman.py -l $(maze) -p SearchAgent -a fn=$(fn) -q

mazes: ## make all mazes
	python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch -q
	python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs -q
	python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=dfs -q

	python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs -q
	python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs -q
	python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=bfs -q

	python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs -q

	python pacman.py -l mediumDottedMaze -p StayEastSearchAgent -q
	python pacman.py -l mediumScaryMaze -p StayWestSearchAgent -q

	python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q

qs: ## make all qs
	for q in q1 q2 q3 q4 ; do \
    	python autograder.py -q $$q ; \
	done

	
help: ## Show help message.
	@printf "Usage:\n"
	@printf "  make $(BLUE)<target>$(NC)\n\n"
	@printf "Targets:\n"
	@perl -nle'print $& if m{^[a-zA-Z0-9_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; \
		{printf "$(BLUE)  %-18s$(NC) %s\n", $$1, $$2}'



##python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch -q
##python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch -q
##python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs -q
##python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=dfs -q
