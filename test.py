def get_answer(IN):
    answer = ""
    split_word = ""
    direction = ["right", "left"]
    angle = ["opposite", "around"]
    times = ["twice", "thrice"]
    
    if "and" in IN.split():
        sub_action_list = [actions.split() for actions in IN.split("and")]
        split_word = "and"
        answer += "Since {}, we consider {} firstly. \n".format(IN, " ".join(sub_action_list[0]))
    elif "after" in IN.split():
        sub_action_list = [actions.split() for actions in IN.split("after")]
        split_word = "after"
        answer += "Since {}, we consider {} firstly. \n".format(IN, " ".join(sub_action_list[1]))
    else:
        sub_action_list = [IN.split()]
        answer += "Let's consider {}. \n".format(" ".join(sub_action_list[0]))
    
    
    for actions in sub_action_list:
        sub_action_sequence = []
        actions_str = " ".join(actions)
        if len(actions) > 4:
            print(f"err:{actions_str}, length")
            continue
        this_times = ""
        this_direction = ""
        this_angle = ""
        if actions[0] == "turn":
            action_kind = 1
        else:
            action_kind = 2
        this_action = actions[0]
        
        if len(actions) > 1:
            if actions[1] in direction:
                this_direction = actions[1]
                if len(actions) == 3:
                    if actions[2] not in times:
                        print(f"err:{actions_str}, times")
                        continue
                    this_times = actions[2]
            elif actions[1] in angle:
                this_angle = actions[1]
                if actions[2] not in direction:
                    print(f"err:{actions_str}, direction")
                    continue
                this_direction = actions[2]
                if len(actions) == 4:
                    if actions[3] not in times:
                        print(f"err:{actions_str}, times")
                        continue
                    this_times = actions[3]
            else:
                print(f"err:{actions_str}, no angle and direction")
                continue
            
                
        if this_direction == "":
            once_action = []
            once_action.append(f"I_{this_action.upper()}")
            answer += "\"{}\" means the agent needs to {}. So, in action sequence is {}. ".format(this_action, this_action, ",".join(once_action))
        elif this_angle == "":
            once_action = []
            answer += f"\"{this_action} {this_direction}\" means the agent needs to turn {this_direction}"
            once_action.append(f"I_TURN_{this_direction.upper()}")
            if action_kind == 2:
                answer += f" and {this_action}"
                once_action.append(f"I_{this_action.upper()}")
            answer += ". "
            answer += "So, in action sequence is {}. ".format(",".join(once_action))
        else:
            once_action = []
            answer += f"\"{this_action} {this_angle} {this_direction}\" means the agent needs to turn {this_direction}"
            once_action.append(f"I_TURN_{this_direction.upper()}")
            if action_kind == 2:
                answer += f" and {this_action}, "
                once_action.append(f"I_{this_action.upper()}")
            else:
                answer += f", "
            if this_angle == "around":
                angle_times = 4
                answer += "and repeat this action sequence four times to complete a 360-degree loop"
            if this_angle == "opposite":
                angle_times = 2
                answer += "and repeat this action sequence two times to complete a 180-degree loop"
            answer += ". "
            print(once_action)
            once_action = once_action * angle_times
            answer += "So, in action sequence is {}. ".format(",".join(once_action))
            
        if this_times != "":
            if this_times == "twice":
                times = 2
            if this_times == "thrice":
                times = 3
            sub_action_sequence = once_action * times
            answer += f"Since we need do {this_times} in command \"{actions_str}\",  this entire sequence is repeated {times} times. "
            answer += "So the In action sequence is {}".format(",".join(sub_action_sequence))        
        else:
            sub_action_sequence = once_action
        answer += "\n"
    
    print(sub_action_list)
    return answer


get_answer("jump around right thrice after walk around right thrice")