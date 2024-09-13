import subprocess

MIDs = ["Cli", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
        "JacksonXml", "Jsoup", "Lang", "Math", "Mockito", "Time", "Closure"]

projects = ["Cli", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
            "JacksonXml", "Jsoup", "Lang", "Math", "Mockito", "Time", "Closure"]


for MID in MIDs:
    print(f"Processing MID: {MID}")

    for project in projects:
        print(f"Running tests for project {project} with MID {MID}")
        subprocess.run(["python", "run_test.py", project, MID])
        subprocess.run(["python", "top_k.py", project, MID])
