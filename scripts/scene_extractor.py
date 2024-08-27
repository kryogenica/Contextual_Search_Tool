
import re

class SceneExtractor:
    def __init__(self, script_location):
        # Load and process the script
        with open(script_location, 'r') as file:
            script_text = file.readlines()
        self.scenes = self.extract_scenes_from_script(script_text)
        self.script_location = script_location

    
    def extract_scenes_from_script(self, script_text):

        scene_list = []
        current_scene_lines = []
        consecutive_blank_lines = 0

        # Regex pattern to identify scene headings with more flexibility
        scene_heading_regex = re.compile(
            r'(INT|EXT|INT/EXT|I/E|INT-EXT|EST|MONTAGE|INT./EXT|I./E|INT.-EXT)'  # Scene type keywords
            r'\.?\s+'                                    # Optional period followed by mandatory space
            r'.*$',                                      # Rest of the heading, including location and time
            re.IGNORECASE                                # Makes matching case-insensitive
        )

        for line in script_text:
            # Ignore lines with only non-alphabetic characters (e.g. page numbers)
            if not line or re.match(r'^[^a-zA-Z]*$', line):
                continue

            # Detect scene headings based on the defined regex pattern
            if scene_heading_regex.match(line.strip()):
                # If a scene is currently being captured, store it before starting a new one
                if current_scene_lines:
                    scene_list.append("\n".join(current_scene_lines))
                    current_scene_lines = []
                # Reset the blank line counter since a new scene is starting
                consecutive_blank_lines = 0

            # Add the current line to the scene being constructed
            current_scene_lines.append(line)

        # If there are any lines left in the last scene, add it to the scene list
        if current_scene_lines:
            scene_list.append("\n".join(current_scene_lines))

        return scene_list


    def scene_distiller(self, scene_num):
        # Chose which scene to distill actions and dialogoues
        one_scene = self.scenes[scene_num]

        # Initialize lists for dialogue and descriptions
        dialogues = []
        descriptions = []

        # Split the scene by lines
        lines = one_scene.split('\n')

        current_description = []
        current_character = None
        current_dialogue_lines = []
        capture_dialogue = False

        for j, line in enumerate(lines):
            # Skip first line in a scene, it should be the header, not needed
            if j == 0:
                continue

            # Strip the line for processing but keep original leading spaces for classification
            stripped_line = line.strip()

            # Skip empty lines
            if not stripped_line:
                continue

            # Description lines have no leading white spaces
            if not line.startswith(" "):
                if capture_dialogue:
                    # Add the current dialogue block to the dialogues list as a dictionary
                    dialogues.append({current_character: '\n'.join(current_dialogue_lines)})
                    capture_dialogue = False
                    current_character = None
                    current_dialogue_lines = []

                # Add line to the current description block
                current_description.append(stripped_line)
            else:
                # Lines with leading spaces indicate character names or dialogue
                leading_spaces = len(line) - len(line.lstrip())

                # Character name lines have more leading spaces than dialogues
                if stripped_line.isupper() and leading_spaces >= 20:  # Assuming 20 spaces for character names
                    if current_description:
                        # Join accumulated description lines into a single string and add to descriptions list
                        descriptions.append(' '.join(current_description))
                        current_description = []

                    if capture_dialogue:
                        # Add the current dialogue block to the dialogues list as a dictionary
                        dialogues.append({current_character: '\n'.join(current_dialogue_lines)})

                    # Start a new dialogue block with the character's name
                    current_character = stripped_line
                    if '(' in current_character:
                        current_character = stripped_line.split('(')[0].strip()
                    current_dialogue_lines = []
                    capture_dialogue = True
                elif capture_dialogue and leading_spaces >= 10:  # Assuming 10 spaces for dialogue
                    # Append dialogue lines under the current character name
                    current_dialogue_lines.append(stripped_line)

        # Add any remaining description or dialogue block
        if current_description:
            descriptions.append('\n'.join(current_description))
        if capture_dialogue and current_character:
            dialogues.append({current_character: ' '.join(current_dialogue_lines)})

        return descriptions, dialogues
