This repository contains the source code that we used to perform experiments in the paper titled "Combining Static Analysis Information and Dynamic Call Graph for Fault Localization".

The structure of this folder is as follows：

- PDGExtractor/: Contains code for extracting Program Dependency Graphs (PDGs) from Java projects.
- DCGExtractor/: Contains code for extracting Dynamic Call Graphs (DCGs) from Java projects.
- Main/: The main experimental code and baselines.

Graph files and datasets are available via the link: https://figshare.com/s/3d7128e46641c1756a4e

## Environment Requirements

Software

- Python == 3.10.6
- PyTorch == 1.12.1+cu113
- CUDA 11.3
- Java Development Kit == 1.8
- Maven == 3.9.6
- Defects4J == 2.0.0

Hardware

- A server with Ubuntu 22.04 OS
- 72GB RAM
- Intel Xeon(R) Platinum 8358P 32-Core CPU @ 2.6 GHz
- 40GB NVIDIA A100 GPU

## PDGExtractor

Here is the extractor for Program Dependency Graphs from `Defects4J` Java source code using static analysis based on `JavaParser`.

You can navigate to the `PDGExtractor` directory, build the project using `Maven,` and then run the extractor (`Jar Package`) with the following command:

```bash
cd PDGExtractor
mvn clean install
java -jar [extractor_location] [project_repository_location + filename] [output_filename]
```

## DCGExtractor

Here is the extractor for Dynamic Call Graphs from `Defects4J` Java source code using `Java Agent` based on `Javassist` (need to adjust its version according to the project Java version of `Defects4J`). It runs as a Java agent and instruments the methods of class files of projects to track their invocations. At JVM exit, prints a table of dynamic caller-callee relationships.

You can navigate to the `DCGExtractor` directory, build the project using `Maven,` and then run the extractor (`Jar Package`) with the following command:

```bash
cd DCGExtractor
mvn clean install
java -Xbootclasspath:/path/to/jre/lib/rt.jar:jar/your-library.jar \
     -javaagent:target/javacg-0.1-SNAPSHOT-dycg-agent.jar="incl=your.package.*;" \
     -jar [project_jar_package] [application_args]
```

## Main

Here is the main experimental code of us as well as the baselines.

Put the graph `pkl` format file in the root directory of the `Main` based on the different methods. Then run `python main.py [project_name]`. For example, `python main.py Codec`. 

You can change the different baseline models and their hyperparameters in the `run.py` file as needed, as well as run cross-project experiments in the `cross.py` file. The scripts will output the `pkl` model file for each method and the corresponding result values. 

The `parse_coverage.py` file is the parser for the results of the coverage information report which is provided by `Defects4J`. You can get these reports by checking out the appropriate project version and using the `defects4j compile` , `defects4j test`  and `defects4j coverage` commands. Refer specifically to the dataset and script (https://defects4j.org/html_doc/d4j/d4j-coverage.html) it provided. The specific model structure and the network of the baselines are in files such as `Model.py` and `GGANN.py`.

