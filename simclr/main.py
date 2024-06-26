import subprocess

def main():
    scripts = [
        "simclr/simCLR+InceptionTime/trainIT.py",
        "simclr/simCLR+InceptionTime/testIT.py",
        "simclr/simCLR+resnet/trainRN.py",
        "simclr/simCLR+resnet/testRN.py",
        "simclr/LR/trainLR.py",
        "simclr/LR/testLR.py",
        "simclr/question.py",
    ]

    # Run each script
    for script in scripts:
        subprocess.run(["python", script])

if __name__ == "__main__":
    main()