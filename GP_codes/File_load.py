import numpy as np

# Some useful constants
deg2rad = np.deg2rad(1)

def load_from_txt(txt_filename):
    '''
    Load rotor performance data from a *.txt file. 
    Parameters:
    -----------
        txt_filename: str
                        Filename of the text containing the Cp, Ct, and Cq data. This should be in the format printed by the write_rotorperformance function
    '''
    print('Loading rotor performace data from text file:', txt_filename)

    with open(txt_filename) as pfile:
        for line in pfile:
            # Read Blade Pitch Angles (degrees)
            if 'Pitch angle' in line:
                pitch_initial = np.array([float(x) for x in pfile.readline().strip().split()])
                pitch_initial_rad = pitch_initial * deg2rad             # degrees to rad            -- should this be conditional?

            # Read Tip Speed Ratios (rad)
            if 'TSR' in line:
                TSR_initial = np.array([float(x) for x in pfile.readline().strip().split()])
            
            # Read Power Coefficients
            if 'Power' in line:
                pfile.readline()
                Cp = np.empty((len(TSR_initial),len(pitch_initial)))
                for tsr_i in range(len(TSR_initial)):
                    Cp[tsr_i] = np.array([float(x) for x in pfile.readline().strip().split()])
            
            # Read Thrust Coefficients
            if 'Thrust' in line:
                pfile.readline()
                Ct = np.empty((len(TSR_initial),len(pitch_initial)))
                for tsr_i in range(len(TSR_initial)):
                    Ct[tsr_i] = np.array([float(x) for x in pfile.readline().strip().split()])

            # Read Torque Coefficients
            if 'Torque' in line:
                pfile.readline()
                Cq = np.empty((len(TSR_initial),len(pitch_initial)))
                for tsr_i in range(len(TSR_initial)):
                    Cq[tsr_i] = np.array([float(x) for x in pfile.readline().strip().split()])

        # return pitch_initial_rad TSR_initial Cp Ct Cq
        # Store necessary metrics for analysis and tuning
        # self.pitch_initial_rad = pitch_initial_rad
        # self.TSR_initial = TSR_initial
        # self.Cp_table = Cp
        # self.Ct_table = Ct 
        # self.Cq_table = Cq
        return pitch_initial_rad, TSR_initial, Cp, Ct, Cq
