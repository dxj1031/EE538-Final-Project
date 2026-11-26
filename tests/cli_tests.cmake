# Qinyan Zeng: numeric end-to-end tests and explicit WILL_FAIL CLI rejections.
set(INT4_CLI_DATA "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
set(INT4_CLI_BAD "${CMAKE_CURRENT_BINARY_DIR}/cli_invalid")
file(MAKE_DIRECTORY "${INT4_CLI_BAD}")
if(Python3_Interpreter_FOUND)
    add_test(NAME int4_rotation_summary_tests COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rotation_summary.py")
    foreach(fixture proposal handoff)
        foreach(mode fp32 direct rotated)
            foreach(kernel scalar blocked)
                foreach(precision float double)
                    add_test(NAME cli_${fixture}_${mode}_${kernel}_${precision}
                        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.py"
                            --exe $<TARGET_FILE:int4_cli> --fixture ${fixture}
                            --mode ${mode} --kernel ${kernel} --precision ${precision})
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endif()

function(int4_cli_reject name x_file w_file)
    add_test(NAME cli_reject_${name} COMMAND int4_cli
        --x "${x_file}" --w "${w_file}"
        --out "${INT4_CLI_BAD}/${name}_Y.txt" --csv "${INT4_CLI_BAD}/${name}.csv"
        --warmups 0 --runs 1 ${ARGN})
    set_tests_properties(cli_reject_${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

foreach(case short trailing nonfinite malformed bad_m bad_k huge)
    if(case STREQUAL "short")
        set(contents "1 2\n1\n")
    elseif(case STREQUAL "trailing")
        set(contents "1 2\n1 -1 extra\n")
    elseif(case STREQUAL "nonfinite")
        set(contents "1 2\nnan 0\n")
    elseif(case STREQUAL "malformed")
        set(contents "one two\n")
    elseif(case STREQUAL "bad_m")
        set(contents "33 2\n")
    elseif(case STREQUAL "bad_k")
        set(contents "1 3\n1 2 3\n")
    else()
        set(contents "18446744073709551615 1024\n")
    endif()
    file(WRITE "${INT4_CLI_BAD}/${case}.txt" "${contents}")
    int4_cli_reject(${case} "${INT4_CLI_BAD}/${case}.txt" "${INT4_CLI_DATA}/proposal_w.txt")
endforeach()
file(WRITE "${INT4_CLI_BAD}/wrong_k.txt" "1 4\n1 2 3 4\n")
file(WRITE "${INT4_CLI_BAD}/bad_n.txt" "2 1025\n")
file(WRITE "${INT4_CLI_BAD}/overflow.txt" "1 2\n1e100 0\n")
int4_cli_reject(wrong_k "${INT4_CLI_BAD}/wrong_k.txt" "${INT4_CLI_DATA}/proposal_w.txt")
int4_cli_reject(bad_n "${INT4_CLI_DATA}/proposal_x.txt" "${INT4_CLI_BAD}/bad_n.txt")
int4_cli_reject(overflow "${INT4_CLI_BAD}/overflow.txt" "${INT4_CLI_DATA}/proposal_w.txt")
int4_cli_reject(missing_file "${INT4_CLI_BAD}/does_not_exist.txt" "${INT4_CLI_DATA}/proposal_w.txt")
foreach(flag mode kernel scale-precision seed)
    int4_cli_reject(bad_${flag} "${INT4_CLI_DATA}/proposal_x.txt" "${INT4_CLI_DATA}/proposal_w.txt"
        --${flag} invalid)
endforeach()
int4_cli_reject(negative_seed "${INT4_CLI_DATA}/proposal_x.txt" "${INT4_CLI_DATA}/proposal_w.txt" --seed -1)
int4_cli_reject(seed_overflow "${INT4_CLI_DATA}/proposal_x.txt" "${INT4_CLI_DATA}/proposal_w.txt"
    --seed 18446744073709551616)
add_test(NAME cli_reject_zero_runs COMMAND int4_cli --x "${INT4_CLI_DATA}/proposal_x.txt"
    --w "${INT4_CLI_DATA}/proposal_w.txt" --out "${INT4_CLI_BAD}/zero_Y.txt"
    --csv "${INT4_CLI_BAD}/zero.csv" --runs 0)
set_tests_properties(cli_reject_zero_runs PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_reject_output_alias COMMAND int4_cli --x "${INT4_CLI_DATA}/proposal_x.txt"
    --w "${INT4_CLI_DATA}/proposal_w.txt" --out "${INT4_CLI_DATA}/proposal_x.txt"
    --csv "${INT4_CLI_BAD}/alias.csv")
set_tests_properties(cli_reject_output_alias PROPERTIES WILL_FAIL TRUE)
