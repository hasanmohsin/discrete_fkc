import os
import torch 
import re 

def parse_a_b(out_resp_txt, invalid_err=1000000.0):
    out_resp = out_resp_txt 
    out_resp_ = out_resp.split("a =")
        
    if len(out_resp_) >= 2:
        out_resp = out_resp_[1]
        out_resp_ = out_resp.split(", and b =")

        # if it can't split on ", and b =", try splitting on ", b = "
        if len(out_resp_) < 2:
            out_resp_ = out_resp.split(", b =")
        # if it still can't split, try splitting on "b = "
        if len(out_resp_) < 2:
            out_resp_ = out_resp.split("b =")
    
        if len(out_resp_) >= 2 and out_resp_[1].count("\n") == 0:
    

            out_a = out_resp_[0]
            out_b = out_resp_[1]

            
            if "," in out_a:
                out_a = out_a.split(",")
                out_a = out_a[0]
            if "/" in out_a:
                out_a = out_a.split("/")
                out_a = out_a[0]
            if "\"" in out_a:
                out_a = out_a.split("\"")
                out_a = out_a[1]
        
            if "," in out_b:
                out_b = out_b.split(",")
                out_b = out_b[0]
            if "/" in out_b:
                out_b = out_b.split("/")
                out_b = out_b[0]
            if "\"" in out_b:
                out_b = out_b.split("\"")
                out_b = out_b[0]
            if "." in out_b:
                out_b = out_b.split(".")
                out_b = out_b[0] + "." + out_b[1]
            

            if "=" in out_b:
                print("out_b: ", out_b)
                out_b = out_b.split("=")
                out_b = out_b[1]
                print("split out_b: ", out_b)
            
            # re
            out_a = re.search(r'-?\d*\.?\d+', out_a).group()
            out_b = re.search(r'-?\d*\.?\d+', out_b).group()

            # convert to float
            out_a = float(out_a)
            out_b = float(out_b)
        else:
                out_a = invalid_err ##-1
                out_b = invalid_err ##-1

    else:
        # try to directly parse numerics
        out_a_re = re.search(r'-?\d*\.?\d+', out_resp)
        if out_a_re is not None:
            out_a = out_a_re.group()
            out_b_txt = out_resp.split(out_a)
            out_b_re = re.search(r'-?\d*\.?\d+', out_b_txt[1])
        else:
            out_a = str(invalid_err)
            out_b_re = None
        
        
        if out_b_re is not None:
            out_b = out_b_re.group()
        else:
            out_b = str(invalid_err)
        
        try:
            out_a = float(out_a)
            out_b = float(out_b)
        
        except:
            out_a = invalid_err
            out_b = invalid_err ##-1

    return out_a, out_b


          
def save_results_to_file(results, filename):
    """
    Save the results to a file.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    torch.save(results, filename)
    print("Results saved to: ", filename)

def save_generated_texts_to_file(generated_texts, filename):
    """
    Save the generated texts to a file.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    with open(filename, 'w') as f:
        print("number of samples: ", len(generated_texts))
        for j in range(len(generated_texts)):
            text = generated_texts[j]
            f.write("\n\nSample {}:".format(j) + '\n')
            f.write(text+'\n')
    return 
