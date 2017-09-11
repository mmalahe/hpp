""" Utilities for producing and processing profiling data.
"""

from subprocess import call
from recordclass import recordclass

Instruction = recordclass("InstrData","name longname kind")
# This database is based on a Haswell processor
instrDB = {
    'vmovss': Instruction("vmovss","Move or Merge Single-Precision Floating-Point Values", "mov"),
    'vinsertps': Instruction("vinsertps","Insert Packed Single Precision Floating-Point Value", "mov"),
    'vfmadd132ps': Instruction("vfmadd132ps","Fused Multiply-Add of Packed Single-Precision Floating-Point Values", "arith"),
    'vfmadd213ps': Instruction("vfmadd213ps","Fused Multiply-Add of Packed Single-Precision Floating-Point Values", "arith"),
    'vfmadd231ps': Instruction("vfmadd321ps","Fused Multiply-Add of Packed Single-Precision Floating-Point Values", "arith"),
    'movslq': Instruction("movslq","Sign-extend 32-bit to 64-bit", "mov"),
    'vinsertf128': Instruction("vinsertf128","Insert Packed Floating-Point Values", "mov"),
    'vmovlhps': Instruction("vmovlhps","Move Packed Single-Precision Floating-Point Values Low to High", "mov"),
    'vmovlps': Instruction("vmovlps","Move Low Packed Single-Precision Floating-Point Values", "mov"),
    'vmovhps': Instruction("vmovlps","Move High Packed Single-Precision Floating-Point Values", "mov"),
    'vpmulld': Instruction("vmulld","Multiply Packed Signed Dword Integers and Store Low Result", "arith"),
    'add': Instruction("add","Add", "arith"),
    'vphaddd': Instruction("vphaddd","Packed Horizontal Add", "arith"),
    'mov': Instruction("mov","Move", "mov"),
    'shl': Instruction("shl","Shift", "arith"),
    'sar': Instruction("sar","Shift", "arith"),
    'lea': Instruction("lea","Load Effective Address", "mov"),
    'vmovq': Instruction("vmovq","Move Quadword", "mov"),
    'vmovdqa': Instruction("vmovdqa","Move Aligned Double Quadword", "mov"),
    'vfmadd132pd': Instruction("vfmadd132pd","Fused Multiply-Add of Packed Double-Precision Floating-Point Values", "arith"),
    'vfmadd213pd': Instruction("vfmadd213pd","Fused Multiply-Add of Packed Double-Precision Floating-Point Values", "arith"),
    'vfmadd231pd': Instruction("vfmadd231pd","Fused Multiply-Add of Packed Double-Precision Floating-Point Values", "arith"),
    'vmovupd': Instruction("vmovupd","Move Unaligned Packed Double-Precision Floating-Point Values", "mov"),
    'vpslld': Instruction("vpslld","Shift Packed Data Left Logical", "arith"),
    'vsusbd': Instruction("vsusbd","Subtract Scalar Double-Precision Floating-Point Values", "arith")
}

def readPerfAnnotated(filename):
    fileIn = open(filename, 'r')
    instrCosts = {}
    lines = fileIn.readlines()
    readStart = False
    for i in range(len(lines)):
        line = lines[i].rstrip("\r\n")
        
        # Actions at boundaries between functions
        isBoundary = line.startswith("----------") and line.endswith("-")
        if isBoundary:
            # On first occurence of boundary, begin reading
            if not readStart:
                readStart = True
                continue
            # Otherwise stop
            else:
                break
                
        # Read
        if readStart:
            linesplit = line.split()
            if len(linesplit) > 1:
                # It contains an instruction
                if linesplit[1] == ":":
                    instrCost = float(linesplit[0])
                    instrName = linesplit[3]
                    if instrCost > 0.0:
                        if instrName in instrCosts.keys():
                            instrCosts[instrName] += instrCost
                        else:
                            instrCosts[instrName] = instrCost
                        
    # Order by decreasing cost
    orderedByCost = sorted(instrCosts, key=instrCosts.get)
    orderedByCost.reverse()
    
    # Determine cost by kind of instruction
    costByKind = {}
    for instr in orderedByCost:
        try:
            instrKind = instrDB[instr].kind
        except:
            print "Type unknown for", instr
            instrKind = 'unknown'
        if instrKind in costByKind.keys():
            costByKind[instrKind] += instrCosts[instr]
        else:
            costByKind[instrKind] = instrCosts[instr]
        print instr, instrCosts[instr]
    
    return costByKind
    
def getPerfPercentages(name=None):
    # Produce output
    annFilename = "annotated.txt"
    args = ["perf","annotate"]
    if name != None:
        args.append(name)
    fileOut = open(annFilename, 'w')
    call(args, stdout=fileOut)
    fileOut.close()
    
    # Parse it
    return readPerfAnnotated(annFilename)
    
    
