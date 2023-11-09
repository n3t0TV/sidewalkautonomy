from importlib.machinery import FrozenImporter
import sys 
sys.path.append("../../../devel/lib/python3/dist-packages/")

import rospy
from  modules.msg import joystick_msg, vp_whereiam_msg
from datetime import datetime
import  numpy as np

vels=[]
dirs=[]
tt=[]

raw_vels=[]
raw_dirs=[]
raw_tt=[]

T0=datetime.now()

def callback_control(data: joystick_msg):

    if data.Autopilot:
        T1 = datetime.now()
        vels.append(data.Speep)
        dirs.append(data.Angle)
        tt.append( (T1-T0).total_seconds()  )

def callback_whereiam(data: vp_whereiam_msg):
        T1 = datetime.now()
        raw_vels.append(data.Speep)
        raw_dirs.append(data.Angle)
        raw_tt.append( (T1-T0).total_seconds()  )
    

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("control_topic", joystick_msg, callback_control)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def save_data(t, v, d, ofile_name):
    data = np.concatenate([t, v, d], axis=1)
    np.save(ofile_name, arr=data)

if __name__ == '__main__':
    listener()

    save_data(t=np.array(tt).reshape(-1,1), 
              v=np.array(vels).reshape(-1,1), 
              d=np.array(dirs).reshape(-1,1), ofile_name="data.nz")
    save_data(t=np.array(raw_tt).reshape(-1,1), 
              v=np.array(raw_vels).reshape(-1,1), 
              d=np.array(raw_dirs).reshape(-1,1), ofile_name="raw_data.nz")