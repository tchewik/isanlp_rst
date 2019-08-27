import functools as fn


def _print_bintree(node, node_info=None, inverted=False, is_top=True):
    # node value string and sub nodes
    string_value, left_node, right_node = node_info(node)
    string_value_width = len(string_value)

    # recurse to sub nodes to obtain line blocks on left and right
    left_text_block = [] if not left_node else _print_bintree(left_node, node_info, inverted, False)
    right_text_block = [] if not right_node else _print_bintree(right_node, node_info, inverted, False)

    # count common and maximum number of sub node lines
    common_lines = min(len(left_text_block), len(right_text_block))
    sublevel_lines = max(len(right_text_block), len(left_text_block))

    # extend lines on shallower side to get same number of lines on both sides
    left_sublines = left_text_block + [""] * (sublevel_lines - len(left_text_block))
    right_sublines = right_text_block + [""] * (sublevel_lines - len(right_text_block))

    # compute location of value or link bar for all left and right sub nodes
    #   * left node's value ends at line's width
    #   * right node's value starts after initial spaces
    left_line_widths = [len(line) for line in left_sublines]
    right_line_widths = [len(line) - len(line.lstrip(" ")) for line in right_sublines]

    # top line value locations, will be used to determine position of current node & link bars
    first_left_width = (left_line_widths + [0])[0]
    first_right_indent = (right_line_widths + [0])[0]

    # width of sub node link under node value (i.e. with slashes if any)
    # aims to center link bars under the value if value is wide enough
    #
    # Value_line:    v     vv    vvvvvv   vvvvv
    # Link_line:    / \   /  \    /  \     / \
    #
    link_spacing = min(string_value_width, 2 - string_value_width % 2)
    left_link_bar = 1 if left_node else 0
    right_link_bar = 1 if right_node else 0
    min_link_width = left_link_bar + link_spacing + right_link_bar
    value_offset = (string_value_width - link_spacing) // 2

    # find optimal position for right side top node
    #   * must allow room for link bars above and between left and right top nodes
    #   * must not overlap lower level nodes on any given line (allow gap of min_spacing)
    #   * can be offset to the left if lower subNodes of right node
    #     have no overlap with subNodes of left node
    min_spacing = 2
    right_node_position = fn.reduce(lambda r, i: max(r, i[0] + min_spacing + first_right_indent - i[1]),
                                    zip(left_line_widths, right_line_widths[0:common_lines]),
                                    first_left_width + min_link_width)

    # extend basic link bars (slashes) with underlines to reach left and right
    # top nodes.
    #
    #        vvvvv
    #       __/ \__
    #      L       R
    #
    link_extra_width = max(0, right_node_position - first_left_width - min_link_width)
    right_link_extra = link_extra_width // 2
    left_link_extra = link_extra_width - right_link_extra

    # build value line taking into account left indent and link bar extension (on left side)
    value_indent = max(0, first_left_width + left_link_extra + left_link_bar - value_offset)
    value_line = " " * max(0, value_indent) + string_value
    slash = "\\" if inverted else "/"
    backslash = "/" if inverted else "\\"
    u_line = "¯" if inverted else "_"

    # build left side of link line
    left_link = "" if not left_node else (" " * first_left_width + u_line * left_link_extra + slash)

    # build right side of link line (includes blank spaces under top node value)
    right_link_offset = link_spacing + value_offset * (1 - left_link_bar)
    right_link = "" if not right_node else (" " * right_link_offset + backslash + u_line * right_link_extra)

    # full link line (will be empty if there are no sub nodes)
    link_line = left_link + right_link

    # will need to offset left side lines if right side sub nodes extend beyond left margin
    # can happen if left subtree is shorter (in he_ight) than right side subtree
    left_indent_width = max(0, first_right_indent - right_node_position)
    left_indent = " " * left_indent_width
    indented_left_lines = [(left_indent if line else "") + line for line in left_sublines]

    # compute distance between left and right sublines based on the_ir value position
    # can be negative if leading spaces need to be removed from right side
    merge_offsets = [len(line) for line in indented_left_lines]
    merge_offsets = [left_indent_width + right_node_position - first_right_indent - w for w in merge_offsets]
    merge_offsets = [p if right_sublines[i] else 0 for i, p in enumerate(merge_offsets)]

    # combine left and right lines using computed offsets
    #   * indented left sub lines
    #   * spaces between left and right lines
    #   * right sub line with extra leading blanks removed.
    merged_sublines = zip(range(len(merge_offsets)), merge_offsets, indented_left_lines)
    merged_sublines = [(i, p, line + (" " * max(0, p))) for i, p, line in merged_sublines]
    merged_sublines = [line + right_sublines[i][max(0, -p):] for i, p, line in merged_sublines]

    # Assemble final result combining
    #  * node value string
    #  * link line (if any)
    #  * merged lines from left and right sub trees (if any)
    tree_lines = [left_indent + value_line] + ([] if not link_line else [left_indent + link_line]) + merged_sublines

    # invert final result if requested
    tree_lines = reversed(tree_lines) if inverted and is_top else tree_lines

    # return intermediate tree lines or print final result
    if is_top:
        print("\n".join(tree_lines))
    else:
        return tree_lines


def print_rst_tree(tree, file):
    def _(n):
        if n.relation:
            value = (n.relation, "%.2f" % n.proba)
        else:
            value = n.text
        return str(value), n.left, n.right

    lines = _print_bintree(_)
    file.write(lines)
