
def select(size, feature_size, avg):
    if (feature_size < 8): 
        if (size <= 6677.0): 
            if (size <= 2404.0): 
                if (feature_size <= 3.0): 
                    if (size <= 1435.0): 
                        if (size <= 1370.5): 
                            pr = 1, 2, 1, 4, 32
                        else: 
                            pr = 1, 2, 1, 2, 32
                        
                    else: 
                        if (feature_size <= 1.5): 
                            pr = 1, 2, 1, 4, 32
                        else: 
                            pr = 1, 2, 1, 4, 32
                        
                else: 
                    if (size <= 1937.0): 
                        if (size <= 1062.5): 
                            pr = 1, 4, 1, 2, 32
                        else: 
                            pr = 1, 4, 1, 2, 32
                        
                    else: 
                        if (size <= 2202.0): 
                            pr = 1, 4, 1, 2, 32
                        else: 
                            pr = 1, 2, 1, 4, 32
            else: 
                if (feature_size <= 1.5): 
                    if (size <= 4729.5): 
                        if (avg <= 30.3): 
                            pr = 1, 2, 1, 4, 32
                        else: 
                            pr = 1, 2, 1, 8, 32
                        
                    else: 
                        if (size <= 6285.5): 
                            pr = 1, 2, 1, 4, 32
                        else: 
                            pr = 1, 2, 1, 4, 32
                        
                    
                else: 
                    if (size <= 3948.0): 
                        if (size <= 2961.0): 
                            pr = 1, 2, 1, 4, 32
                        else: 
                            pr = 1, 2, 1, 4, 32
                        
                    else: 
                        if (avg <= 38.33): 
                            pr = 1, 4, 1, 4, 32
                        else: 
                            pr = 1, 4, 1, 4, 32
                        
        else: 
            if (avg <= 1.15): 
                if (feature_size <= 3.0): 
                    if (size <= 703065.0): 
                        if (size <= 351532.5): 
                            pr = 1, 1, 2, 4, 32
                        else: 
                            pr = 2, 1, 2, 8, 32
                        
                    else: 
                        if (feature_size <= 1.5): 
                            pr = 1, 1, 2, 8, 32
                        else: 
                            pr = 1, 2, 2, 2, 32
                        
                    
                else: 
                    if (size <= 1406130.0): 
                        if (avg <= 1.15): 
                            pr = 2, 2, 2, 1, 16
                        else: 
                            pr = 1, 4, 4, 1, 16
                        
                    else: 
                        pr = 2, 2, 4, 1, 16
                    
                
            else: 
                if (feature_size <= 1.5): 
                    if (size <= 359204.0): 
                        if (avg <= 7.19): 
                            pr = 1, 1, 1, 8, 32
                        else: 
                            pr = 1, 1, 1, 8, 32
                        
                    else: 
                        if (size <= 2188236.0): 
                            pr = 1, 1, 2, 8, 32
                        else: 
                            pr = 1, 1, 2, 8, 32
                        
                    
                else: 
                    if (avg <= 35.71): 
                        if (size <= 70469.0): 
                            pr = 1, 2, 1, 4, 32
                        else: 
                            pr = 4, 1, 2, 4, 32
                        
                    else: 
                        if (size <= 331916.5): 
                            pr = 2, 2, 1, 8, 32
                        else: 
                            pr = 4, 1, 2, 4, 32
    else: 
        if (feature_size <= 48.0): 
            if (feature_size <= 24.0): 
                if (size <= 20974.5): 
                    if (avg <= 9.39): 
                        if (size <= 7155.0): 
                            sr = 1, 32, 4, 8
                        else: 
                            sr = 1, 32, 4, 8
                        
                    else: 
                        if (avg <= 35.43): 
                            sr = 1, 16, 4, 8
                        else: 
                            sr = 1, 16, 4, 8
                        
                    
                else: 
                    if (feature_size <= 12.0): 
                        if (size <= 93022.5): 
                            sr = 1, 8, 4, 8
                        else: 
                            sr = 1, 8, 8, 8
                        
                    else: 
                        if (size <= 247962.5): 
                            sr = 1, 16, 8, 8
                        else: 
                            sr = 2, 8, 8, 8
                        
                    
                
            else: 
                if (size <= 55061.5): 
                    if (size <= 6521.0): 
                        if (size <= 6413.5): 
                            sr = 1, 32, 4, 8
                        else: 
                            sr = 1, 64, 8, 4
                        
                    else: 
                        if (avg <= 502.41): 
                            sr = 1, 32, 4, 8
                        else: 
                            sr = 1, 64, 8, 4
                        
                    
                else: 
                    if (avg <= 4.22): 
                        if (size <= 2351251.0): 
                            sr = 2, 16, 8, 8
                        else: 
                            sr = 2, 16, 32, 8
                        
                    else: 
                        if (size <= 220246.0): 
                            sr = 2, 16, 8, 8
                        else: 
                            sr = 2, 16, 8, 8
                        
                    
                
            
        else: 
            if (feature_size <= 96.0): 
                if (size <= 10487.0): 
                    if (avg <= 1.15): 
                        sr = 1, 32, 4, 8
                    else: 
                        if (avg <= 38.97): 
                            sr = 1, 64, 4, 8
                        else: 
                            sr = 1, 64, 4, 8
                        
                    
                else: 
                    if (size <= 1094118.0): 
                        if (avg <= 492.02): 
                            sr = 2, 32, 8, 8
                        else: 
                            sr = 2, 32, 16, 8
                        
                    else: 
                        if (size <= 1127504.0): 
                            sr = 2, 32, 32, 8
                        else: 
                            sr = 2, 32, 16, 8
                        
                    
                
            else: 
                if (size <= 18918.5): 
                    if (size <= 5341.5): 
                        if (size <= 3142.5): 
                            sr = 1, 64, 4, 8
                        else: 
                            sr = 2, 64, 4, 4
                        
                    else: 
                        if (avg <= 36.3): 
                            sr = 2, 32, 8, 4
                        else: 
                            sr = 2, 64, 8, 4
                        
                    
                else: 
                    if (size <= 747442.0): 
                        if (avg <= 492.8): 
                            sr = 2, 64, 8, 4
                        else: 
                            sr = 2, 64, 16, 4
                        
                    else: 
                        if (avg <= 15.04): 
                            sr = 2, 64, 16, 4
                        else: 
                            sr = 2, 64, 8, 4
                        
    if feature_size >= 8:
        return sr
    else:
        return pr
        