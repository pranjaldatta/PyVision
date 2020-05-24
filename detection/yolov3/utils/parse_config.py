def parse_config(path):
    """
    This method parses a config file and constructs a list of blocks.

    Each block is a singular unit in the architecture as explained in 
    the paper. Blocks are represented as a dictionary in the list.

    Input: 
    - path: path to the config file.

    Returns:
    - a list containing a dictionary of individual block information.
    """
    cfg_file = open(path, "r")

    lines = cfg_file.read().split("\n")
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if line[0] != '#']
    lines = [line.strip() for line in lines]
    
    block = {}
    blocks_list = []

    for line in lines: 
        if line[0] == "[":
            if len(block) != 0:
                blocks_list.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            idx, value = line.split("=")
            block[idx.rstrip()] = value.lstrip()
    blocks_list.append(block)

    return blocks_list






