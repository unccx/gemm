#!/usr/bin/env python
import lldb


def print_mat(debugger, command, exe_ctx, result, internal_dict):

    frame = exe_ctx.GetFrame()
    matrix_view = frame.EvaluateExpression(command)
    if not matrix_view.IsValid() or matrix_view.GetError().Fail():
        result.PutCString("Invalid expression.")
        return

    # 获取 matrix_view 的成员变量
    rows = matrix_view.GetChildMemberWithName("rows").GetValueAsUnsigned()
    cols = matrix_view.GetChildMemberWithName("cols").GetValueAsUnsigned()
    start_row = matrix_view.GetChildMemberWithName("start_row").GetValueAsUnsigned()
    start_col = matrix_view.GetChildMemberWithName("start_col").GetValueAsUnsigned()
    lda = matrix_view.GetChildMemberWithName("lda").GetValueAsUnsigned()
    data_ptr = matrix_view.GetChildMemberWithName("data")

    result.PutCString(f"rows:{rows}")
    result.PutCString(f"cols:{cols}")
    result.PutCString(f"start_row:{start_row}")
    result.PutCString(f"start_col:{start_col}")
    result.PutCString(f"lda:{lda}")

    def index(i, j):
        """根据StorageLayout和二维索引[i, j]计算一维索引"""
        type_name = matrix_view.GetType().GetName()
        if "RowMajor" in type_name or "(GEMM::StorageLayout)0" in type_name:
            return (start_row + i) * lda + (start_col + j)
        elif "ColumnMajor" in type_name or "(GEMM::StorageLayout)1" in type_name:
            return (start_col + j) * lda + (start_row + i)

    def get_element(error, index):
        pointee_type_name = data_ptr.GetType().GetPointeeType().GetName()
        data = data_ptr.GetPointeeData(index)
        get_func = {
            "signed char": data.GetSignedInt8,
            "int": data.GetSignedInt32,
            "double": data.GetDouble,
            "float": data.GetFloat,
        }
        return get_func[pointee_type_name](error, 0)

    # 打印矩阵
    error = lldb.SBError()
    matrix = ""
    result.PutCString("matrix:")
    for i in range(rows):
        for j in range(cols):
            value = get_element(error, index(i, j))
            if error.Fail():
                result.PutCString("Unable to read memory: " + error.GetCString())
                return
            matrix += str(value) + f"[{index(i, j)}]" + "\t"
        matrix += "\n"
    result.PutCString(matrix)


# (lldb) command script import scripts/lldb_matrix.py
# 将会调用 __lldb_init_module() 函数
def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("command script add -f print_matrix.print_mat print_mat")
