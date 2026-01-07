/*
 * cut_delphes_tree.cpp
 *
 * Utility to skim large Delphes ROOT files and produce small sample ROOT files
 * for rapid testing and development of the ML pipeline.
 *
 * This tool is optional and is not part of the core training/inference workflow.
 * It is used only to generate compact samples from full-statistics simulations.
 *
 * Requires ROOT.
 */

#include <iostream>
#include <string>
#include <memory>

#include "TFile.h"
#include "TKey.h"
#include "TTree.h"
#include "TList.h"
#include "TH1.h"
#include "TIterator.h"

static void copy_all_other_objects(TFile* fin, TFile* fout, const std::string& tree_name) {
    // Copy everything except the main tree (histograms, etc.) if present.
    TIter nextkey(fin->GetListOfKeys());
    TKey* key;
    while ((key = (TKey*)nextkey())) {
        if (key->GetName() == tree_name) continue;

        TKey* highest = fin->GetKey(key->GetName());
        if (key != highest) continue;

        TObject* obj = key->ReadObj();
        if (!obj) continue;

        if (obj->InheritsFrom(TH1::Class())) {
            ((TH1*)obj)->SetDirectory(fout); 
        }

        fout->cd();
        obj->Write(key->GetName(), TObject::kOverwrite);
        delete obj;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " input.root output.root N [tree_name]\n";
        std::cerr << "Default tree_name = Delphes\n";
        return 2;
    }

    std::string inpath = argv[1];
    std::string outpath = argv[2];
    long long N = std::stoll(argv[3]);
    std::string tree_name = (argc >= 5) ? argv[4] : "Delphes";

    std::unique_ptr<TFile> fin(TFile::Open(inpath.c_str(), "READ"));
    if (!fin || fin->IsZombie()) {
        std::cerr << "ERROR: cannot open input file: " << inpath << "\n";
        return 1;
    }

    TTree* tin = fin->Get<TTree>(tree_name.c_str());
    if (!tin) {
        std::cerr << "ERROR: cannot find tree '" << tree_name << "' in " << inpath << "\n";
        fin->Close();
        return 1;
    }

    std::unique_ptr<TFile> fout(TFile::Open(outpath.c_str(), "RECREATE"));
    if (!fout || fout->IsZombie()) {
        std::cerr << "ERROR: cannot create output file: " << outpath << "\n";
        fin->Close();
        return 1;
    }

    copy_all_other_objects(fin.get(), fout.get(), tree_name);

    tin->SetBranchStatus("*", 0);            // Deactivate all branches
    tin->SetBranchStatus("Jet*", 1);         // Re-activate all branches starting with "Jet"
    tin->SetBranchStatus("Event.Weight", 1); // Re-activate weight
    tin->SetBranchStatus("Event.CrossSection", 1); // Re-activate cross section

    long long nentries = tin->GetEntries();
    long long ncopy = (N < 0) ? nentries : std::min(N, nentries);

    fout->cd();
    TTree* tout = tin->CloneTree(ncopy);  

    if (tout){
        tout->Write();
    }

    fout->Close();

    std::cout << "Wrote " << ncopy << " entries: " << outpath << "\n";
    return 0;
}

