import ephemeral as eph
import newisolatedcode as nicode
import sys
import os
import subprocess
from colorama import Fore, Back, Style

def main():
    #check_installed_packages('requirements.txt')
    # print a nice greeting message with decorations and colors
    print(Back.GREEN + Fore.WHITE + "Necessary python packages present" + Style.RESET_ALL)
    print(Fore.YELLOW + "This program will either run ephemeral.py or newisolatedcode.py depending on your input." + Style.RESET_ALL)
    print(Fore.CYAN + "Please enter the training path and receiver port as command-line arguments." + Style.RESET_ALL)
    print(Fore.MAGENTA + "************************************************************" + Style.RESET_ALL)
    # check the length of the sys.args
    while True: # loop until valid input is given
        try:
            if len(sys.argv) != 3: # if the number of arguments is not 3
                raise ValueError("Invalid number of arguments. Please enter exactly two arguments.")
            else: # if the number of arguments is 3
                # read input from sys.args <training path and receiver port>
                training_path = sys.argv[1]
                receiver_port = sys.argv[2]

                # check if the training path contains a file named model.pkl
                if os.path.isfile(os.path.join(training_path, "model.pkl")):
                    eph.main(training_path)
                else:
                    
                    nicode.main()
                break # exit the loop
        except ValueError as e: # catch the ValueError exception
            print(Fore.RED + str(e) + Style.RESET_ALL) # print the error message in red color
            print(Fore.CYAN + "Please re-enter the arguments in the correct format." + Style.RESET_ALL) # ask the user to re-enter the arguments
            sys.argv = [sys.argv[0]] + input().split() # read the new input from the user and assign it to sys.argv

if __name__ == "__main__":
    main()
def check_installed_packages(requirements_file):
    try:
        with open(requirements_file, 'r') as file:
            packages = [line.strip() for line in file.readlines()]

        not_installed = []
        for package in packages:
            result = subprocess.run(["pip", "show", package], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                not_installed.append(package)

        if not_installed:
            print("The following packages are not installed:")
            for package in not_installed:
                print(package)
        else:
            print("All packages in requirements.txt are installed.")

    except FileNotFoundError:
        print(f"File not found: {requirements_file}")

if __name__ == "__main__":
    requirements_file = "requirements.txt"  # Change to the path of your requirements.txt file
    #check_installed_packages(requirements_file)

