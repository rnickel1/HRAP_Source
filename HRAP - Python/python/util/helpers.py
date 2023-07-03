"""
helpers.py

Extra helper function for conversions and units that are not available
in other packages
"""


# Converts total impulse to motor class and some percentage
# MATLAB equivalent; impulse.m
def impulse_to_class_percent(impulse):
    if impulse <= 1.25:
        return "A", 100 * impulse / 1.25
    elif impulse <= 5:
        return "B", 100 * (impulse - 1.25) / 2.5
    elif impulse <= 10:
        return "C", 100 * (impulse - 5) / 5
    elif impulse <= 20:
        return "D", 100 * (impulse - 10) / 10
    elif impulse <= 40:
        return "E", 100 * (impulse - 20) / 20
    elif impulse <= 80:
        return "F", 100 * (impulse - 40) / 40
    elif impulse <= 160:
        return "G", 100 * (impulse - 80) / 80
    elif impulse <= 320:
        return "H", 100 * (impulse - 160) / 160
    elif impulse <= 640:
        return "I", 100 * (impulse - 320) / 320
    elif impulse <= 1280:
        return "J", 100 * (impulse - 640) / 640
    elif impulse <= 2560:
        return "K", 100 * (impulse - 1280) / 1280
    elif impulse <= 5120:
        return "L", 100 * (impulse - 2560) / 2560
    elif impulse <= 10240:
        return "M", 100 * (impulse - 5120) / 5120
    elif impulse <= 20480:
        return "N", 100 * (impulse - 10240) / 10240
    elif impulse <= 40960:
        return "O", 100 * (impulse - 20480) / 20480
    elif impulse <= 81920:
        return "P", 100 * (impulse - 40960) / 40960
    elif impulse <= 163840:
        return "Q", 100 * (impulse - 81920) / 81920
    elif impulse <= 327680:
        return "R", 100 * (impulse - 163840) / 163840
    elif impulse <= 655360:
        return "S", 100 * (impulse - 327680) / 327680
    elif impulse <= 1310720:
        return "T", 100 * (impulse - 655360) / 655360
    else:
        return None, None
