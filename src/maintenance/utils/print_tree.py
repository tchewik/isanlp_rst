import functools as fn


def printBTree(node, nodeInfo=None, inverted=False, isTop=True):
    # node value string and sub nodes
    info = nodeInfo(node)
    
    if info:
        stringValue, leftNode, rightNode = info

        stringValueWidth  = len(stringValue)

        # recurse to sub nodes to obtain line blocks on left and right
        leftTextBlock     = [] if not leftNode else printBTree(leftNode, nodeInfo, inverted, False)
        rightTextBlock    = [] if not rightNode else printBTree(rightNode, nodeInfo, inverted, False)

        # count common and maximum number of sub node lines
        commonLines       = min(len(leftTextBlock),len(rightTextBlock))
        subLevelLines     = max(len(rightTextBlock),len(leftTextBlock))

        # extend lines on shallower side to get same number of lines on both sides
        leftSubLines      = leftTextBlock  + [""] *  (subLevelLines - len(leftTextBlock))
        rightSubLines     = rightTextBlock + [""] *  (subLevelLines - len(rightTextBlock))

        # compute location of value or link bar for all left and right sub nodes
        #   * left node's value ends at line's width
        #   * right node's value starts after initial spaces
        leftLineWidths    = [ len(line) for line in leftSubLines  ]                            
        rightLineIndents  = [ len(line)-len(line.lstrip(" ")) for line in rightSubLines ]

        # top line value locations, will be used to determine position of current node & link bars
        firstLeftWidth    = (leftLineWidths   + [0])[0]  
        firstRightIndent  = (rightLineIndents + [0])[0] 

        # width of sub node link under node value (i.e. with slashes if any)
        # aims to center link bars under the value if value is wide enough
        # 
        # ValueLine:    v     vv    vvvvvv   vvvvv
        # LinkLine:    / \   /  \    /  \     / \ 
        #
        linkSpacing       = min(stringValueWidth, 2 - stringValueWidth % 2)
        leftLinkBar       = 1 if leftNode  else 0
        rightLinkBar      = 1 if rightNode else 0
        minLinkWidth      = leftLinkBar + linkSpacing + rightLinkBar
        valueOffset       = (stringValueWidth - linkSpacing) // 2

        # find optimal position for right side top node
        #   * must allow room for link bars above and between left and right top nodes
        #   * must not overlap lower level nodes on any given line (allow gap of minSpacing)
        #   * can be offset to the left if lower subNodes of right node 
        #     have no overlap with subNodes of left node                                                                                                                                 
        minSpacing        = 2
        rightNodePosition = fn.reduce(lambda r,i: max(r,i[0] + minSpacing + firstRightIndent - i[1]), \
                                     zip(leftLineWidths,rightLineIndents[0:commonLines]), \
                                     firstLeftWidth + minLinkWidth)

        # extend basic link bars (slashes) with underlines to reach left and right
        # top nodes.  
        #
        #        vvvvv
        #       __/ \__
        #      L       R
        #
        linkExtraWidth    = max(0, rightNodePosition - firstLeftWidth - minLinkWidth )
        rightLinkExtra    = linkExtraWidth // 2
        leftLinkExtra     = linkExtraWidth - rightLinkExtra

        # build value line taking into account left indent and link bar extension (on left side)
        valueIndent       = max(0, firstLeftWidth + leftLinkExtra + leftLinkBar - valueOffset)
        valueLine         = " " * max(0,valueIndent) + stringValue
        slash             = "\\" if inverted else  "/"
        backslash         = "/" if inverted else  "\\"
        uLine             = "¯" if inverted else  "_"

        # build left side of link line
        leftLink          = "" if not leftNode else ( " " * firstLeftWidth + uLine * leftLinkExtra + slash)

        # build right side of link line (includes blank spaces under top node value) 
        rightLinkOffset   = linkSpacing + valueOffset * (1 - leftLinkBar)                      
        rightLink         = "" if not rightNode else ( " " * rightLinkOffset + backslash + uLine * rightLinkExtra )

        # full link line (will be empty if there are no sub nodes)                                                                                                    
        linkLine          = leftLink + rightLink

        # will need to offset left side lines if right side sub nodes extend beyond left margin
        # can happen if left subtree is shorter (in height) than right side subtree                                                
        leftIndentWidth   = max(0,firstRightIndent - rightNodePosition) 
        leftIndent        = " " * leftIndentWidth
        indentedLeftLines = [ (leftIndent if line else "") + line for line in leftSubLines ]

        # compute distance between left and right sublines based on their value position
        # can be negative if leading spaces need to be removed from right side
        mergeOffsets      = [ len(line) for line in indentedLeftLines ]
        mergeOffsets      = [ leftIndentWidth + rightNodePosition - firstRightIndent - w for w in mergeOffsets ]
        mergeOffsets      = [ p if rightSubLines[i] else 0 for i,p in enumerate(mergeOffsets) ]

        # combine left and right lines using computed offsets
        #   * indented left sub lines
        #   * spaces between left and right lines
        #   * right sub line with extra leading blanks removed.
        mergedSubLines    = zip(range(len(mergeOffsets)), mergeOffsets, indentedLeftLines)
        mergedSubLines    = [ (i,p,line + (" " * max(0,p)) )       for i,p,line in mergedSubLines ]
        mergedSubLines    = [ line + rightSubLines[i][max(0,-p):]  for i,p,line in mergedSubLines ]                        

        # Assemble final result combining
        #  * node value string
        #  * link line (if any)
        #  * merged lines from left and right sub trees (if any)
        treeLines = [leftIndent + valueLine] + ( [] if not linkLine else [leftIndent + linkLine] ) + mergedSubLines

        # invert final result if requested
        treeLines = reversed(treeLines) if inverted and isTop else treeLines

        # return intermediate tree lines or print final result
        if isTop : return "\n".join(treeLines)
        else     : return treeLines        

def print_rst_tree(tree, file):
    def _(n):
        if n.relation != 'elementary':
            value = (n.relation, n.proba)
        else:
            value = n.text
        
        return str(value), n.left, n.right

    lines = printBTree(tree, _)
    file.write(lines)
