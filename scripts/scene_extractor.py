
import re

class SceneExtractor:
    def __init__(self, script_text):
        self.script_text = script_text

    def extract_scenes_from_script(self):
        scene_list = []
        current_scene_lines = []
        for line in self.script_text:
            if 'INT.' in line or 'EXT.' in line:
                if current_scene_lines:
                    scene_list.append(''.join(current_scene_lines))
                    current_scene_lines = []
            current_scene_lines.append(line)
        if current_scene_lines:
            scene_list.append(''.join(current_scene_lines))
        return scene_list

    def separate_scene(self, scene):
        dialogues = []
        descriptions = []
        lines = scene.splitlines()
        for line in lines:
            if re.match(r'^[A-Z ]+$', line.strip()):
                dialogues.append(line.strip())
            else:
                descriptions.append(line.strip())
        return descriptions, dialogues
