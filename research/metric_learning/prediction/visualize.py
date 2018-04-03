# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.spatial
import os, sys
import numpy as np
import optparse
from collections import defaultdict

def get_checked(anno_dir):
    qname2cnames = defaultdict(set)
    for filename in os.listdir(anno_dir):
        qname = '_'.join(filename.split('_')[1:])+'.jpg'
        for line in open(os.path.join(anno_dir, filename), 'r'):
            cname = '_'.join(line.strip().split('_')[1:])+'.jpg'
            qname2cnames[qname].add(cname)
    return qname2cnames

def extend_query(q_name, q_embed, qname2cnames, cname2embeds):
    q_names = [q_name]
    q_embeds = [q_embed]
    if q_name in qname2cnames:
        cname_list = list(qname2cnames[q_name])
        q_names.extend(cname_list)
        for name in cname_list:
            q_embeds.append(cname2embeds[name])
    return q_names, np.array(q_embeds)

def get_neareast_result(q_embeds, c_embeds, q_names, c_names, filtered_names, top_k = 20):
    dist = scipy.spatial.distance.cdist(q_embeds, c_embeds, 'cosine')
    ordered_indices = dist.argsort()
    neareast_results = []
    record = [0]*dist.shape[0]
    names = set()
    while True:
        min_dist = 10000.0
        min_i = -1
        min_j = -1
        for i in xrange(dist.shape[0]):
            j = ordered_indices[i,record[i]] # for dist
            if (c_names[j] in names) or (c_names[j] in filtered_names):
                record[i] += 1
                continue
            if min_dist > dist[i,j]:
                min_dist = dist[i,j]
                min_i = i
                min_j = j

        if min_i > -1:
            names.add(c_names[min_j])
            neareast_results.append(c_names[min_j])
            record[min_i] += 1
            if len(neareast_results) == top_k:
                break
    return neareast_results

def main():
    parser = optparse.OptionParser()
    parser.add_option(
        '-e',
        '--embedding_file',
        help="the path of embedding file",
        type='str',
        default=
        '/raid/data/nice/metric_data/checked_logs/0004/info/embeddings.txt'
    )
    parser.add_option(
        '-t',
        '--template_file',
        help="the path of template file",
        type='str',
        default=
        '/raid/data/nice/metric_data/checked_logs/0004/info/index.html'
    )
    parser.add_option(
        '-a',
        '--anno_dir',
        help="the path of template file",
        type='str',
        default=
        '/raid/data/nice/metric_data/checked_logs/0004/anno'
    )

    opts, args = parser.parse_args()

    html = '''
    <head><meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <script src="https://cdn.bootcss.com/jquery/1.12.4/jquery.min.js"></script>
    <script>
        function change_opacity(query_id){
            var query_checkbox = $("#checkbox_"+query_id);
            if (query_checkbox.is(":checked")) {
                document.getElementById('img_'+query_id).style.opacity=1.0;
                query_checkbox.checked = false;
            }
            else {
                document.getElementById('img_'+query_id).style.opacity=0.5;
                query_checkbox.checked = true;
            }
        }
        function submit_checked(query_id){
            var number = query_id;
            $('input:checkbox[name='+query_id+']:checked').each(function(k){
                number += ','+$(this).val();
            })
            $.post("/checked_names", {'names':number});
            alert('ok');
        }
    </script>
    </head>
  '''
    
    # read embeds from embedding file
    query_embeds = []
    query_names = []

    cand_embeds = []
    cand_names = []
    
    with open(opts.embedding_file, 'r') as f:
        for line in f:
            [name, label, embedding] = line.strip().split('\t')
            label = int(label)
            embedding = [float(emb) for emb in embedding.split(',')]
            if label == 1:
                query_names.append(name)
                query_embeds.append(embedding)
            elif label == 2:
                cand_names.append(name)
                cand_embeds.append(embedding)
            else:
                raise ValueError

    cname2embeds = {}
    for name,embeds in zip(cand_names, cand_embeds):
        cname2embeds[name] = embeds

    query_embeds = np.array(query_embeds)
    cand_embeds = np.array(cand_embeds)
    top_k = 20

    # get mumu's annotation
    qname2cnames = get_checked(opts.anno_dir)

    # filter all checked candidates
    filtered_names = set()
    for names in qname2cnames.values():
        filtered_names |= names

    for i, qname in enumerate(query_names):
        # only show the checked query
        # if qname not in qname2cnames:
        #     continue

        # extend query with checked candidates
        q_names, q_embeds = extend_query(qname, query_embeds[i,:], qname2cnames, cname2embeds)

        # get top k nearest candidates
        results = get_neareast_result(q_embeds, cand_embeds, q_names, cand_names, filtered_names, top_k)
        query_name = str(i) + '_' + qname.split('.')[0]
        html += '<img height=256 width=256 src=http://10.8.10.37:7777/' + qname + '>'
        html += '\n'
        ct = 0
        for j in xrange(top_k):
            cand_name = str(i) + '_' + results[j].split('.')[0]
            html += '<label for="checkbox_' + cand_name + '">'
            html += '\n'
            html += '<img id="img_' + cand_name + '" height=256 width=256 src=http://10.8.10.37:7777/' + results[j]
            html += ' onclick="change_opacity(\'' + cand_name + '\')"' + '>'
            html += '\n'
            html += '</label>'
            html += '\n'
            html += '<input type="checkbox" id="checkbox_' + cand_name + '" name="' + query_name + '" style="display:none" value="' + cand_name + '">'
            html += '\n'
            
        html += '<input type="submit" id="submit_' + query_name + '" onclick="submit_checked(\'' + query_name + '\')">'
        html += '<br />'
        html += '\n'

    with open(opts.template_file, 'w') as fw:
        fw.write(html)


if __name__ == '__main__':
    main()
