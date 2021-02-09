import pandas as pd
import numpy as np

import operator

def build_graph_covid(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    essentials.append(first_instance_date)
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} ".format("first case", "observed", first_instance_date))
    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        if idx%2 == 0:
            if wave != "":
                wave_list.append(wave)
            wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} ".format(content + ' ' + wave_enum[idx] + " wave", "lasted", start_time + " to " + end_time)
        else:
            wave += "<H> {} <R> {} <T> {} ".format(wave_enum[idx] + " wave", "lasted", start_time + " to " + end_time)
    wave_list.append(wave)
    for w in wave_list:
        graph.append(w)
    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(trend_type)
        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type, start_time + " to " + end_time)
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format("cases" , trend_type, start_time + " to " + end_time)
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    essentials.append(str(peak))
    essentials.append(peak_time)
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak value", str(peak) + ' at ' + peak_time)
    graph.append(peak_text)
    return graph, essentials       

def build_graph_covid_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    essentials.append(first_instance_date)
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} ".format("first case", "observed", first_instance_date))
    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        if idx%2 == 0:
            if wave != "":
                wave_list.append(wave)
            wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} ".format(content + ' ' + wave_enum[idx] + " wave", "lasted", start_time + " to " + end_time)
        else:
            wave += "<H> {} <R> {} <T> {} ".format(wave_enum[idx] + " wave", "lasted", start_time + " to " + end_time)
    wave_list.append(wave)
    for w in wave_list:
        graph.append(w)
    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_end = raw_time_series[end]
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(trend_type)
        essentials.append(str(val_at_end))
        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format("cases" , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    essentials.append(str(peak))
    essentials.append(peak_time)
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak value", str(peak) + ' at ' + peak_time)
    graph.append(peak_text)
    return graph, essentials


import operator
def build_graph_covid_form1(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    first_val = int(raw_time_series[0])
    current_val = int(raw_time_series[-1])
    current_date = original_data.iloc[[-1]]['month'].values[0] + ' ' + str(original_data.iloc[[-1]]['year'].values[0])

    total = np.sum(raw_time_series)

    if first_val > current_val:
        overall = "decreasing"
    elif first_val < current_val:
        overall = "increasing"
    else:
        overall = "remained same"

    
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content + " steadily increasing", "location", location) + "<H> {} <R> {} <T> {} ".format(content.split(' ')[0] + " first case", "observed", first_instance_date)
        + "<H> {} <R> {} <T> {}".format("Total cases reported to date", int(total), "by " + current_date))

    essentials.append(first_instance_date.split(' ')[1])
    essentials.append(int(total))
    essentials.append(current_date.split(' ')[1])

    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    
    graph.append("<H> {} <R> {} <T> {} <H> {} <R> {} <T> {}".format(content.split(' ')[0], 'invaded', location, content.split(' ')[0], 'with', str(num_waves) + ' waves'))
    for idx in range(num_waves):
        wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        wave_peak = max(raw_time_series[start:end])
        
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {} ".format(content.split(' ')[0], wave_enum[idx].lower() + " wave", "lasted from " + start_time + " to " + end_time, wave_enum[idx].lower() + " wave", "reported daily peak", int(wave_peak))
        else:
            if idx % 2 != 0:
                wave += "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {} ".format(wave_enum[idx] + " wave", "reported daily peak", int(wave_peak), wave_enum[idx] + " wave", content.split(' ')[0], "lasted from " + start_time + " to " + end_time)
            else:
                wave += "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {} ".format(wave_enum[idx] + " wave", "reported daily peak", int(wave_peak), wave_enum[idx] + " wave", "duration", "lasted from " + start_time + " to " + end_time)
        
        essentials.append(start_time.split(' ')[1])
        essentials.append(end_time.split(' ')[1])
        essentials.append(int(wave_peak))

        graph.append(wave)
    
    #Trends
    trend_text = ""
    trend_list = []
    first_time = False
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        
        val_at_end = int(raw_time_series[end])
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        
        essentials.append(val_at_end)
        essentials.append(start_time.split(' ')[1])
        essentials.append(end_time.split(' ')[1])

        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , start_time + " to " + end_time, trend_type.lower() + ' to ' + str(val_at_end))
            first_time = True
        else:
            if first_time == True:
                trend_text += "<H> {} <R> {} <T> {} ".format("cases" , start_time + " to " + end_time, trend_type.lower() + ' to ' + str(val_at_end))
                first_time = False
            else:
                trend_text += "<H> {} <R> {} <T> {} ".format("cases" , trend_type.lower(), start_time + " to " + end_time + ' to ' + str(val_at_end))
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    
    essentials.append(int(peak))
    essentials.append(peak_time.split(' ')[1])
    
    peak_text += "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {} ".format("Highest noted occurance" , content + " as of " + peak_time, "stands at " + str(int(peak)), "Highest noted occurance", "location", location)
    graph.append(peak_text)
    return graph, essentials
    
def build_template_covid(content, location, wave_data, trend_data, original_data, raw_time_series):
    template = ""
    
    #Intro
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    template = content + " in " + location + " was first seen in {}. ".format(first_instance_date)
    
    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        template += "The {} wave lasted from {} to {}. ".format(wave_enum[idx], start_time, end_time)
    
    #Trends
    for trend in trend_data:
        start = trend[0]
        end = trend[1]
        trend_type = trend[2].lower()
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        template += "The cases {} from {} to {}. ".format(trend_type, start_time, end_time)
 
    #Peak
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    template += "The peak value is {} at {}.".format(str(peak), peak_time)
    
    return template

def build_template_covid_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    template = ""
    
    #Intro
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    current_date = original_data.iloc[[-1]]['month'].values[0] + ' ' + str(original_data.iloc[[-1]]['year'].values[0])
    total = np.sum(raw_time_series)
    template += "The number of Coronavirus cases in " + location + " has been steadily increasing, with the first case being observed in {} ".format(first_instance_date)
    template += "and the total cases reported to date is {} by {}. ".format(int(total), current_date)
    
    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        wave_peak = int(max(raw_time_series[start:end]))
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        if idx == 0:
            template += "The {} wave of Coronavirus lasted from {} to {} and had a reported daily peak of {}".format(wave_enum[idx], start_time, end_time, wave_peak)
        elif idx < (num_waves - 1):
            template += ", the {} wave of Coronavirus lasted from {} to {} and had a reported daily peak of {}".format(wave_enum[idx], start_time, end_time, wave_peak)
        else:
            template += ", and the {} wave of Coronavirus lasted from {} to {} and had a reported daily peak of {}. ".format(wave_enum[idx], start_time, end_time, wave_peak)
    
    #Trends
    for idx, trend in enumerate(trend_data):
        start = trend[0]
        end = trend[1]
        trend_type = trend[2].lower()

        val_at_end = int(raw_time_series[end])

        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])

        if idx == 0:
            template += "Coronavirus cases {} to {} from {} to {}".format(trend_type, val_at_end, start_time, end_time)
        elif idx < (len(trend_data) - 1):
            template += ", Coronavirus cases {} to {} from {} to {}".format(trend_type, val_at_end, start_time, end_time)
        else:
            template += ", and Coronavirus cases {} to {} from {} to {}. ".format(trend_type, val_at_end, start_time, end_time)
 
    #Peak
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    template += "The peak value of Coronavirus cases as of {} is {}.".format(peak_time, str(int(peak)))
    
    return template


def build_graph_exports_form1(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    first_val = raw_time_series[0]
    current_val = raw_time_series[-1]
    current_date = original_data.iloc[[-1]]['month'].values[0] + ' ' + str(original_data.iloc[[-1]]['year'].values[0])

    total = np.sum(raw_time_series)

    if first_val > current_val:
        overall = "decrease"
    elif first_val < current_val:
        overall = "increase"
    else:
        overall = "remained same"

    
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format("value of " + content.lower() + " steady " + overall, "location", location) + "<H> {} <T> {} ".format("first record", first_instance_date))

    essentials.append(first_instance_date.split(' ')[1])

    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    
    graph.append("<H> {} <R> {} <T> {}".format(content.split(' ')[1] + " growth can be classified","into " + str(num_waves) + " seasons", "in " + location))

    for idx in range(num_waves):
        wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        wave_peak = round(max(raw_time_series[start:end]),2)

        essentials.append(wave_peak)
        essentials.append(start_time.split(' ')[1])
        essentials.append(end_time.split(' ')[1])

        if idx == 0:
            wave += "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {} ".format(wave_enum[idx] + " season of " + content.lower(), "lasted" , start_time + " to " + end_time, wave_enum[idx] + " season", "monthly exports high as", wave_peak)
        else:
            wave += "<H> {} <R> {} <T> {} <R> {} <T> {} ".format(wave_enum[idx] + " season", "spanned", start_time + " to " + end_time, "the " + wave_enum[idx].lower() + " monthly exports high as", wave_peak)
        graph.append(wave)
    
    #Trends
    trend_list = []
    first_time = False
    for idx, trend in enumerate(trend_data):
        trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_start = round(raw_time_series[start],2)
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        
        essentials.append(val_at_start)
        essentials.append(val_at_end)
        essentials.append(start_time.split(' ')[1])
        essentials.append(end_time.split(' ')[1])


        trend_text += "<H> {} <R> {} <T> {} ".format("The " + content.split(' ')[1].lower() , trend_type.lower() + ' from ' + str(val_at_start) + ' to ' + str(val_at_end), start_time + " to " + end_time)
        trend_list.append(trend_text)
        
        
    for t in trend_list:
        graph.append(t)
    
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    peak = round(peak,2)

    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    
    essentials.append(peak)
    essentials.append(peak_time.split(' ')[1])

    peak_text += "<H> {} <R> {} <T> {} ".format("Greatest monthly " + content.split(' ')[1].lower(), "noted" , "at " + peak_time + " to be " + str(peak))
    graph.append(peak_text)
    return graph, essentials

def build_graph_exports_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    essentials.append(first_instance_date)
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} ".format(content + " first data", "observed", first_instance_date))
    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        if idx%2 == 0:
            if wave != "":
                wave_list.append(wave)
            wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} ".format(content + ' ' + wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
        else:
            wave += "<H> {} <R> {} <T> {} ".format(wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
    wave_list.append(wave)
    for w in wave_list:
        graph.append(w)
    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_end = raw_time_series[end]
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(trend_type)
        essentials.append(str(val_at_end))
        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format("number of " + content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    essentials.append(str(peak))
    essentials.append(peak_time)
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak value", str(peak) + ' at ' + peak_time)
    graph.append(peak_text)
    return graph, essentials

def build_template_exports_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    template = ""
    
    #Intro
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    template += "The value of merchandise exports has increased steadily in " + location + " since {}. ".format(first_instance_date)

    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        wave_peak = round(max(raw_time_series[start:end]),2)

        if idx == 0 and (num_waves-1) != 0:
            template += "The peak value of the {} season of exports from {} to {} was {}".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)
        elif idx == 0 and (num_waves-1) == 0:
            template += "The peak value of the {} season of exports from {} to {} was {}. ".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)
        elif idx < (num_waves - 1):
            template += ", the peak value of the {} season of exports from {} to {} was {}".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)
        else:
            template += ", and the peak value of the {} season of exports from {} to {} was {}. ".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)
    
    #Trends
    for idx, trend in enumerate(trend_data):
        start = trend[0]
        end = trend[1]
        trend_type = trend[2].lower()
        val_at_start = round(raw_time_series[start],2)
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        
        if idx == 0:
            template += "The exports {} from {} to {} from {} to {}".format(trend_type, val_at_start, val_at_end, start_time, end_time)
        elif idx < (len(trend_data)-1):
            template += ", the exports {} from {} to {} from {} to {}".format(trend_type, val_at_start, val_at_end, start_time, end_time)
        else:
            template += ", and the exports {} from {} to {} from {} to {}. ".format(trend_type, val_at_start, val_at_end, start_time, end_time)
    
    #Peak
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    peak = round(peak,2)
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    template += "The highest montly exports was observed in {} to be {}.".format(peak_time, str(peak))
    
    return template

def build_graph_poll_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    essentials.append(first_instance_date)
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} ".format(content + " first data", "observed", first_instance_date))
    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        if idx%2 == 0:
            if wave != "":
                wave_list.append(wave)
            wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} ".format(content + ' ' + wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
        else:
            wave += "<H> {} <R> {} <T> {} ".format(wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
    if wave != "":
        wave_list.append(wave)
    for w in wave_list:
        graph.append(w)
    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(trend_type)
        essentials.append(str(val_at_end))
        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    essentials.append(str(peak))
    essentials.append(peak_time)
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak value", str(peak) + ' at ' + peak_time)
    graph.append(peak_text)
    return graph, essentials

def build_template_poll_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    template = ""
    
    #Intro
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    
    template += "The first data for mean carbon monoxide in " + location + " was observed in {}. ".format(first_instance_date)
    
    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        wave_peak = round(max(raw_time_series[start:end]),2)
        if idx == 0:
            template += "The {} cycle lasted from {} to {} with a peak of {}".format(wave_enum[idx], start_time, end_time, wave_peak)
        elif idx < (num_waves -1):
            template += ", the {} cycle lasted from {} to {} with a peak of {}".format(wave_enum[idx], start_time, end_time, wave_peak)
        else:
            template += ", and the {} cycle lasted from {} to {} with a peak of {}. ".format(wave_enum[idx], start_time, end_time, wave_peak)
    
    #Trends
    for idx, trend in enumerate(trend_data):
        start = trend[0]
        end = trend[1]
        trend_type = trend[2].lower()
        val_at_start = round(raw_time_series[start],2)
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        if idx == 0:
            template += "The observed values {} from {} to {} from {} to {}".format(trend_type, val_at_end, val_at_start, start_time, end_time)
        elif idx < (len(trend_data)-1):
            template += ", the observed values {} from {} to {} from {} to {}".format(trend_type, val_at_end, val_at_start, start_time, end_time)
        else:
            template += ", and the observed values {} from {} to {} from {} to {}. ".format(trend_type, val_at_end, val_at_start, start_time, end_time)
    
    #Peak
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    peak = round(peak, 2)
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    template += "The highest value of mean carbon monoxide was observed in {} at {}.".format(peak_time, str(peak))
    
    return template

def build_graph_polls_form1(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    first_val = raw_time_series[0]
    current_val = raw_time_series[-1]
    current_date = original_data.iloc[[-1]]['month'].values[0] + ' ' + str(original_data.iloc[[-1]]['year'].values[0])

    total = np.sum(raw_time_series)

    if first_val > current_val:
        overall = "decrease"
    elif first_val < current_val:
        overall = "increase"
    else:
        overall = "remained same"

    
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content + " first data", "was observed", first_instance_date) + "<H> {} <R> {} <T> {} ".format(content, "location", location))

    essentials.append(first_instance_date.split(' ')[1])

    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    
    graph.append("<H> {} <R> {} <T>".format(content + " observed can be dissected", "into " + str(num_waves) + " regimes of interest", "in " + location))
    
    for idx in range(num_waves):
        wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        wave_avg = round(np.mean(raw_time_series[start:end]),2)

        if idx == 0:
            wave += "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {} ".format(wave_enum[idx] + " regime of " + content.lower(), "lasted" , start_time + " to " + end_time, wave_enum[idx] + " regime", "average", wave_avg)
        else:
            wave += "<H> {} <R> {} <T> {} <R> {} <T> {} ".format(wave_enum[idx] + " regime", "spanned", start_time + " to " + end_time, "the average", wave_avg)
        
        essentials.append(wave_avg)
        essentials.append(start_time.split(' ')[1])
        essentials.append(end_time.split(' ')[1])

        graph.append(wave)
    

    #Trends
    trend_text = ""
    trend_list = []
    first_time = False
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_start = round(raw_time_series[start],2)
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        
        essentials.append(val_at_start)
        essentials.append(val_at_end)
        essentials.append(start_time.split(' ')[1])
        essentials.append(end_time.split(' ')[1])

        trend_text += "<H> {} <R> {} <T> {} ".format("The observed values", trend_type.lower() + ' from ' + str(val_at_end) + ' to ' + str(val_at_start), start_time + " to " + end_time)
        
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    peak = round(peak,2)
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    essentials.append(peak)
    essentials.append(peak_time.split(' ')[1])
    peak_text += "<H> {} <R> {} <T> {} ".format(content + " emissions", "peaked" , peak_time + " at " + str(peak))
    graph.append(peak_text)
    return graph, essentials

def build_graph_gtemp_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    essentials.append(first_instance_date)
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} ".format(content + " first data", "observed", first_instance_date))
    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        if idx%2 == 0:
            if wave != "":
                wave_list.append(wave)
            wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} ".format(content + ' ' + wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
        else:
            wave += "<H> {} <R> {} <T> {} ".format(wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
    if wave != "":
        wave_list.append(wave)
    for w in wave_list:
        graph.append(w)
    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(trend_type)
        essentials.append(str(val_at_end))
        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    essentials.append(str(peak))
    essentials.append(peak_time)
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak value", str(round(peak,2)) + ' at ' + peak_time)
    graph.append(peak_text)
    return graph, essentials

def build_template_gtemp_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    template = ""
    
    #Intro
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = original_data.iloc[[first_instance]]['month'].values[0] + ' ' + str(original_data.iloc[[first_instance]]['year'].values[0])
    template += "The average template in " + location + " was first measured in {}. ".format(first_instance_date)

    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        wave_peak = round(max(raw_time_series[start:end]),2)
        if end == len_data:
            end -= 1
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])
        
        if idx == 0:
            template += "The {} cycle lasted from {} to {} with a peak of {}".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)
        elif idx < (num_waves-1):
            template += ", the {} cycle lasted from {} to {} with a peak of {}".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)
        else:
            template += ", and the {} cycle lasted from {} to {} with a peak of {}. ".format(wave_enum[idx].lower(), start_time, end_time, wave_peak)

    #Trends
    for idx, trend in enumerate(trend_data):
        start = trend[0]
        end = trend[1]
        trend_type = trend[2].lower()
        val_at_end = round(raw_time_series[end],2)
        start_time = original_data.iloc[[start]]['month'].values[0] + ' ' + str(original_data.iloc[[start]]['year'].values[0])
        end_time = original_data.iloc[[end]]['month'].values[0] + ' ' + str(original_data.iloc[[end]]['year'].values[0])

        if idx == 0:
            template += "The temperature from {} to {} {} to {}".format(start_time, end_time, trend_type, val_at_end)
        elif idx < (len(trend_data)-1):
            template += ", the temperature from {} to {} {} to {}".format(start_time, end_time, trend_type, val_at_end)
        else:
            template += ", and the temperature from {} to {} {} to {}. ".format(start_time, end_time, trend_type, val_at_end)
 
    #Peak
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = original_data.iloc[[peak_index]]['month'].values[0] + ' ' + str(original_data.iloc[[peak_index]]['year'].values[0])
    template += "The average temperature reached its peak at {} in {}.".format(str(round(peak,2)), peak_time)
    
    return template

def build_graph_gtemp_form1(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = str(original_data.iloc[[first_instance]]['year'].values[0])
    
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {}".format("first measurement", "taken", first_instance_date))
    
    essentials.append(first_instance_date)

    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = str(original_data.iloc[[start]]['year'].values[0])
        end_time = str(original_data.iloc[[end]]['year'].values[0])
        wave_peak = round(max(raw_time_series[start:end]),2)
        
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(wave_peak)

        wave = "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {}".format(wave_enum[idx] + ' ' + content.split(' ')[0], "cycle", "lasted from " + start_time + " to " + end_time, wave_enum[idx].lower() + ' cycle', "peaked at", str(wave_peak) + ' celsius')
        wave_list.append(wave)
    
    for w in wave_list:
        graph.append(w)

    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_start = round(raw_time_series[start],2)
        val_at_end = round(raw_time_series[end],2)
        start_time = str(original_data.iloc[[start]]['year'].values[0])
        end_time = str(original_data.iloc[[end]]['year'].values[0])
        
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(val_at_end)

        trend_text += "<H> {} <R> {} <T> {} ".format("temperature", "from " + start_time + " to " + end_time, trend_type + ' to ' + str(val_at_end) + ' celsius')
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    peak = round(peak, 2)
    if peak_index == len_data:
        peak_index -= 1
    peak_time = str(original_data.iloc[[peak_index]]['year'].values[0])
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak observed value", 'in ' + peak_time + ' at ' + str(peak))
    
    essentials.append(peak_time)
    essentials.append(peak)

    graph.append(peak_text)
    return graph, essentials


def build_graph_pop_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = str(original_data.iloc[[first_instance]]['Year'].values[0])
    essentials.append(first_instance_date)
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} ".format(content + " first data", "reported", first_instance_date))
    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        if idx%2 == 0:
            if wave != "":
                wave_list.append(wave)
            wave = ""
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = str(original_data.iloc[[start]]['Year'].values[0])
        end_time = str(original_data.iloc[[end]]['Year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        if idx == 0:
            wave += "<H> {} <R> {} <T> {} ".format(content + ' ' + wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
        else:
            wave += "<H> {} <R> {} <T> {} ".format(wave_enum[idx] + " cycle", "lasted", start_time + " to " + end_time)
    if wave != "":
        wave_list.append(wave)
    for w in wave_list:
        graph.append(w)
    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_end = round(raw_time_series[end],2)
        start_time = str(original_data.iloc[[start]]['Year'].values[0])
        end_time = str(original_data.iloc[[end]]['Year'].values[0])
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(trend_type)
        essentials.append(str(val_at_end))
        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , trend_type + ' to ' + str(val_at_end), start_time + " to " + end_time)
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = str(original_data.iloc[[peak_index]]['Year'].values[0])
    essentials.append(str(peak))
    essentials.append(peak_time)
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak value", str(round(peak,2)) + ' at ' + peak_time)
    graph.append(peak_text)
    return graph, essentials

def build_graph_pop_form1(content, location, wave_data, trend_data, original_data, raw_time_series):
    essentials = []
    graph = []
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = str(original_data.iloc[[first_instance]]['Year'].values[0])
    
    #Intro
    graph.append("<H> {} <R> {} <T> {} ".format(content, "location", location) + "<H> {} <R> {} <T> {} <H> {} <T> {}".format(content, "first recorded", first_instance_date, content, "Steadily Increasing"))
    essentials.append(first_instance_date)

    #Waves
    wave = ""
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    wave_list = []
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = str(original_data.iloc[[start]]['Year'].values[0])
        end_time = str(original_data.iloc[[end]]['Year'].values[0])
        wave_peak = int(max(raw_time_series[start:end]))
        
        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(wave_peak)
        
        wave = "<H> {} <R> {} <T> {} <H> {} <R> {} <T> {}".format(wave_enum[idx] + ' ' + content.split(' ')[0], "boom", "lasted from " + start_time + " to " + end_time, wave_enum[idx].lower() + ' boom', "peaked at", wave_peak)
        wave_list.append(wave)
    
    for w in wave_list:
        graph.append(w)

    #Trends
    trend_text = ""
    trend_list = []
    for idx, trend in enumerate(trend_data):
        if idx%2 == 0:
            if trend_text != "":
                trend_list.append(trend_text)
            trend_text = ""
        start = trend[0]
        end = trend[1]
        if end == len_data:
            end -= 1
        trend_type = trend[2].lower()
        val_at_start = int(raw_time_series[start])
        val_at_end = int(raw_time_series[end])
        start_time = str(original_data.iloc[[start]]['Year'].values[0])
        end_time = str(original_data.iloc[[end]]['Year'].values[0])

        essentials.append(start_time)
        essentials.append(end_time)
        essentials.append(val_at_start)
        essentials.append(val_at_end)

        if idx == 0:
            trend_text += "<H> {} <R> {} <T> {} ".format(content , "from " + start_time + " to " + end_time, trend_type + ' from ' + str(val_at_start) + ' to ' + str(val_at_end))
        else:
            trend_text += "<H> {} <R> {} <T> {} ".format(content.split(' ')[1] , "from " + start_time + " to " + end_time, trend_type + ' from ' + str(val_at_start) + ' to ' + str(val_at_end))
        
    trend_list.append(trend_text)
    for t in trend_list:
        graph.append(t)
    
    #Peak
    peak_text = ""
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    peak = int(peak)
    if peak_index == len_data:
        peak_index -= 1
    peak_time = str(original_data.iloc[[peak_index]]['Year'].values[0])
    peak_text += "<H> {} <R> {} <T> {} ".format(content , "peak observed value", 'in ' + peak_time + ' at ' + str(peak))
    graph.append(peak_text)

    essentials.append(peak)
    essentials.append(peak_time)
    return graph, essentials

def build_template_pop_nums(content, location, wave_data, trend_data, original_data, raw_time_series):
    template = ""
    
    #Intro
    first_instance = next((i for i, x in enumerate(raw_time_series) if x), None)
    first_instance_date = str(original_data.iloc[[first_instance]]['Year'].values[0])

    template += "The population in " + location + " is steadily increasing and was first recorded in {}. ".format(first_instance_date)

    #Waves
    wave_enum = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eigth", "Ninth", "Tenth"]
    num_waves = len(wave_data)
    len_data = len(original_data)
    for idx in range(num_waves):
        start = wave_data[idx][0]
        end = wave_data[idx][1]
        if end == len_data:
            end -= 1
        start_time = str(original_data.iloc[[start]]['Year'].values[0])
        end_time = str(original_data.iloc[[end]]['Year'].values[0])
        wave_peak = max(raw_time_series[start:end])

        if idx == 0:
            template += "The {} population boom lasted from {} to {} and peaked at {}".format(wave_enum[idx], start_time, end_time, wave_peak)
        elif idx < (num_waves-1):
            template += ", the {} population boom lasted from {} to {} and peaked at {}".format(wave_enum[idx], start_time, end_time, wave_peak)
        else:
            template += ", and the {} population boom lasted from {} to {} and peaked at {}. ".format(wave_enum[idx], start_time, end_time, wave_peak)
    
    #Trends
    for idx, trend in enumerate(trend_data):
        start = trend[0]
        end = trend[1]
        trend_type = trend[2].lower()
        val_at_start = int(raw_time_series[start])
        val_at_end = int(raw_time_series[end])
        start_time = str(original_data.iloc[[start]]['Year'].values[0])
        end_time = str(original_data.iloc[[end]]['Year'].values[0])

        if idx == 0:
            template += "The population from {} to {} {} from {} to {}".format(start_time, end_time, trend_type, val_at_start, val_at_end)
        elif idx < (len(trend_data)-1):
            template += ", the population from {} to {} {} from {} to {}".format(start_time, end_time, trend_type, val_at_start, val_at_end)
        else:
            template += ", and the population from {} to {} {} from {} to {}. ".format(start_time, end_time, trend_type, val_at_start, val_at_end)
 
    #Peak
    peak_index, peak = max(enumerate(raw_time_series), key=operator.itemgetter(1))
    if peak_index == len_data:
        peak_index -= 1
    peak_time = str(original_data.iloc[[peak_index]]['Year'].values[0])
    template += "The peak observed value for population was {} at {}.".format(int(peak), peak_time)
    
    return template
