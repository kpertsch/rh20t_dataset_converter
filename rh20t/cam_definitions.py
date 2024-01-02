
# mapping from cam IDs to semantic camera name per robot config
CFG_TO_CAM = dict(
    cfg1=dict(
        overhead='035622060973',
        front='750612070851',
        side_left='039422060546',
        side_right='750612070853',
        wrist='043322070878',
        shoulder='038522063145',
    ),
    cfg2=dict(
        overhead='104122061850',
        front='037522062165',
        side_left='104122063678',
        side_right='105422061350',
        wrist='104422070042',
        shoulder='036422060215',
    ),
    cfg3=dict(
        overhead='104422070011',
        front='038522062288',
        side_left='104122062823',   # this is actually "shoulder right”
        side_right='104122062295',
        wrist='045322071843',
        shoulder='036422060909',
    ),
    cfg4=dict(
        overhead='104422070011',
        front='038522062288',
        side_left='104122062823',   # this is actually "shoulder right”
        side_right='104122062295',
        wrist='045322071843',
        shoulder='036422060909',
    ),
    cfg5=dict(
        overhead='104122061850',
        front='037522062165',
        side_left='104122063678',
        side_right='105422061350',
        wrist='135122079702',
        shoulder='036422060215',
    ),
    cfg6=dict(
        overhead='104122060811',
        front='104122061018',       # this should be “front slightly right”
        side_left='104122064161',
        side_right='104122063633',
        wrist='135122070361',
        shoulder='104122061602',    # “shoulder right”
    ),
    cfg7=dict(
        overhead='104122060811',
        front='104122061018',       # this should be “front slightly right”
        side_left='104122064161',
        side_right='104122063633',
        wrist='135122070361',
        shoulder='104122061602',    # “shoulder right”
    ),
)